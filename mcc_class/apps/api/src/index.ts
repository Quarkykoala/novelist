import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createClient } from '@supabase/supabase-js';
import { buildContentHash, buildVerificationResponse, generateIssuancePdf, normalizeTagIds } from './letter-utils';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
    throw new Error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY for dev) must be set.');
}

const supabase = createClient(supabaseUrl, supabaseKey);

app.set('trust proxy', true);
app.use(cors());
app.use(express.json());

// --- Master Lists ---

app.get('/api/departments', async (req: Request, res: Response) => {
    const { context } = req.query;
    const query = supabase.from('departments').select('*');
    if (context) {
        query.eq('context', String(context));
    }
    const { data, error } = await query;

    if (error) return res.status(500).json({ error: error.message });
    res.json(data);
});

app.get('/api/tags', async (req: Request, res: Response) => {
    const { context } = req.query;
    const query = supabase.from('tags').select('*');
    if (context) {
        query.eq('context', String(context));
    }
    const { data, error } = await query;

    if (error) return res.status(500).json({ error: error.message });
    res.json(data);
});

// --- Letters ---

app.post('/api/letters', async (req: Request, res: Response) => {
    const { id, context, department_id, tag_ids, content, created_by } = req.body;
    const source_ip = req.ip || '0.0.0.0';

    if (!content || !created_by) {
        return res.status(400).json({ error: 'content and created_by are required.' });
    }

    if (id) {
        const { data: currentLetter, error: letterError } = await supabase
            .from('letters')
            .select('id, context, department_id, status')
            .eq('id', id)
            .single();

        if (letterError || !currentLetter) {
            return res.status(404).json({ error: 'Letter not found.' });
        }

        if (context && context !== currentLetter.context) {
            return res.status(400).json({ error: 'Context cannot be changed for an existing letter.' });
        }

        if (currentLetter?.status === 'APPROVED' || currentLetter?.status === 'ISSUED') {
            return res.status(409).json({ error: 'Cannot update an approved or issued letter. Create a new draft instead.' });
        }

        const updateData: Record<string, string> = {};
        if (department_id && department_id !== currentLetter.department_id) {
            updateData.department_id = department_id;
        }

        if (Object.keys(updateData).length > 0) {
            const { error: updateError } = await supabase
                .from('letters')
                .update({ ...updateData, updated_at: new Date().toISOString() })
                .eq('id', id);
            if (updateError) return res.status(500).json({ error: updateError.message });
        }

        const tagIdsProvided = tag_ids !== undefined;
        if (tagIdsProvided && !Array.isArray(tag_ids)) {
            return res.status(400).json({ error: 'tag_ids must be an array of strings.' });
        }

        let finalTagIds: string[] = [];
        if (tagIdsProvided) {
            finalTagIds = normalizeTagIds(tag_ids);
            const { error: deleteError } = await supabase.from('letter_tags').delete().eq('letter_id', id);
            if (deleteError) return res.status(500).json({ error: deleteError.message });
            if (finalTagIds.length > 0) {
                const tagsToInsert = finalTagIds.map((tagId: string) => ({ letter_id: id, tag_id: tagId }));
                const { error: tagError } = await supabase.from('letter_tags').insert(tagsToInsert);
                if (tagError) return res.status(500).json({ error: tagError.message });
            }
        } else {
            const { data: existingTags, error: tagReadError } = await supabase
                .from('letter_tags')
                .select('tag_id')
                .eq('letter_id', id);
            if (tagReadError) return res.status(500).json({ error: tagReadError.message });
            finalTagIds = (existingTags || []).map((row) => row.tag_id);
        }

        // Get latest version number
        const { data: versions, error: versionError } = await supabase
            .from('letter_versions')
            .select('version_number')
            .eq('letter_id', id)
            .order('version_number', { ascending: false })
            .limit(1);

        if (versionError) return res.status(500).json({ error: versionError.message });

        const nextVersion = (versions?.[0]?.version_number || 0) + 1;
        const finalDepartmentId = (updateData.department_id as string) || currentLetter.department_id;
        const contentHash = buildContentHash({
            letterId: id,
            versionNumber: nextVersion,
            context: currentLetter.context,
            departmentId: finalDepartmentId,
            tagIds: finalTagIds,
            content
        });

        // Insert new version
        const { data: version, error: vError } = await supabase
            .from('letter_versions')
            .insert([{
                letter_id: id,
                version_number: nextVersion,
                content,
                content_hash: contentHash,
                created_by
            }])
            .select()
            .single();

        if (vError) return res.status(500).json({ error: vError.message });

        // Log update
        await supabase.from('audit_logs').insert([{
            actor_id: created_by,
            action: 'UPDATE',
            entity_type: 'LETTER',
            entity_id: id,
            metadata: { version_number: nextVersion },
            source_ip
        }]);

        return res.json({ letter_id: id, version });
    }

    if (!context || !department_id) {
        return res.status(400).json({ error: 'context and department_id are required.' });
    }

    // 1. Create Letter Base
    const { data: letter, error: lError } = await supabase
        .from('letters')
        .insert([{ context, department_id, created_by, status: 'DRAFT' }])
        .select()
        .single();

    if (lError) return res.status(500).json({ error: lError.message });

    // 2. Assign Tags
    const normalizedTagIds = normalizeTagIds(tag_ids);
    if (normalizedTagIds.length > 0) {
        const tagsToInsert = normalizedTagIds.map((tagId: string) => ({ letter_id: letter.id, tag_id: tagId }));
        const { error: tError } = await supabase.from('letter_tags').insert(tagsToInsert);
        if (tError) return res.status(500).json({ error: tError.message });
    }

    // 3. Create initial version
    const contentHash = buildContentHash({
        letterId: letter.id,
        versionNumber: 1,
        context: letter.context,
        departmentId: letter.department_id,
        tagIds: normalizedTagIds,
        content
    });

    const { data: version, error: vError } = await supabase
        .from('letter_versions')
        .insert([{
            letter_id: letter.id,
            version_number: 1,
            content,
            content_hash: contentHash,
            created_by
        }])
        .select()
        .single();

    if (vError) return res.status(500).json({ error: vError.message });

    // Log creation
    await supabase.from('audit_logs').insert([{
        actor_id: created_by,
        action: 'CREATE',
        entity_type: 'LETTER',
        entity_id: letter.id,
        metadata: { context, department_id },
        source_ip
    }]);

    res.status(201).json({ letter, version });
});

app.post('/api/letters/:id/approve', async (req: Request, res: Response) => {
    const { id } = req.params;
    const { approver_id, comment } = req.body;
    const source_ip = req.ip || '0.0.0.0';

    if (!approver_id) {
        return res.status(400).json({ error: 'approver_id is required.' });
    }

    // 1. Get latest version
    const { data: version, error: versionError } = await supabase
        .from('letter_versions')
        .select('id')
        .eq('letter_id', id)
        .order('version_number', { ascending: false })
        .limit(1)
        .single();

    if (versionError) return res.status(500).json({ error: versionError.message });
    if (!version) return res.status(404).json({ error: 'No versions found' });

    // 2. Insert Approval
    const { error: aError } = await supabase
        .from('approvals')
        .insert([{ letter_version_id: version.id, approver_id, comment }]);

    if (aError) return res.status(500).json({ error: aError.message });

    // 3. Update Letter Status
    const { error: uError } = await supabase
        .from('letters')
        .update({ status: 'APPROVED', updated_at: new Date().toISOString() })
        .eq('id', id);

    if (uError) return res.status(500).json({ error: uError.message });

    // 4. Audit Log
    await supabase.from('audit_logs').insert([{
        actor_id: approver_id,
        action: 'APPROVE',
        entity_type: 'LETTER',
        entity_id: id,
        metadata: { version_id: version.id, comment },
        source_ip
    }]);

    res.json({ message: 'Letter approved successfully' });
});

app.get('/api/committees', async (req: Request, res: Response) => {
    const { context } = req.query;
    const query = supabase.from('committees').select('*');
    if (context) {
        query.eq('context', String(context));
    }
    const { data, error } = await query;
    if (error) return res.status(500).json({ error: error.message });
    res.json(data);
});

app.post('/api/letters/:id/committee-approve', async (req: Request, res: Response) => {
    const { id } = req.params;
    const { committee_id, approver_id } = req.body;
    const source_ip = req.ip || '0.0.0.0';

    if (!committee_id || !approver_id) {
        return res.status(400).json({ error: 'committee_id and approver_id are required.' });
    }

    // 1. Get latest version
    const { data: version, error: versionError } = await supabase
        .from('letter_versions')
        .select('id')
        .eq('letter_id', id)
        .order('version_number', { ascending: false })
        .limit(1)
        .single();

    if (versionError) return res.status(500).json({ error: versionError.message });
    if (!version) return res.status(404).json({ error: 'No versions found' });

    // 2. Insert Committee Approval
    const { error: cError } = await supabase
        .from('committee_approvals')
        .insert([{ committee_id, letter_version_id: version.id, approver_id }]);

    if (cError) return res.status(500).json({ error: cError.message });

    // 3. Update Letter Status
    const { error: uError } = await supabase
        .from('letters')
        .update({ status: 'APPROVED', updated_at: new Date().toISOString() })
        .eq('id', id);

    if (uError) return res.status(500).json({ error: uError.message });

    // 4. Audit Log
    await supabase.from('audit_logs').insert([{
        actor_id: approver_id,
        action: 'COMMITTEE_APPROVE',
        entity_type: 'LETTER',
        entity_id: id,
        metadata: { committee_id, version_id: version.id },
        source_ip
    }]);

    res.json({ message: 'Committee approval recorded and letter approved' });
});

app.post('/api/letters/:id/issue', async (req: Request, res: Response) => {
    const { id } = req.params;
    const { issued_by, channel, printer_id } = req.body;
    const source_ip = req.ip || '0.0.0.0';

    if (!issued_by || !channel) {
        return res.status(400).json({ error: 'issued_by and channel are required.' });
    }

    // 1. Check if approved
    const { data: letter, error: lError } = await supabase
        .from('letters')
        .select('*, departments(name)')
        .eq('id', id)
        .single();

    if (lError || !letter) return res.status(404).json({ error: 'Letter not found' });
    if (letter.status !== 'APPROVED') return res.status(400).json({ error: 'Only approved letters can be issued' });

    // 2. Get latest version
    const { data: version, error: versionError } = await supabase
        .from('letter_versions')
        .select('*')
        .eq('letter_id', id)
        .order('version_number', { ascending: false })
        .limit(1)
        .single();

    if (versionError) return res.status(500).json({ error: versionError.message });
    if (!version) return res.status(404).json({ error: 'No version found' });

    // 3. Generate QR Payload (Verification URL)
    const verificationBaseUrl = process.env.VERIFICATION_BASE_URL || 'http://localhost:5173/verify';
    const verificationUrl = `${verificationBaseUrl.replace(/\/$/, '')}/${version.content_hash}`;
    const pdfBase64 = await generateIssuancePdf({
        context: letter.context,
        departmentName: letter.departments?.name,
        content: version.content,
        contentHash: version.content_hash,
        verificationUrl
    });

    // 4. Record Issuance
    const { data: issuance, error: iError } = await supabase
        .from('issuances')
        .insert([{
            letter_version_id: version.id,
            issued_by,
            channel,
            qr_payload: verificationUrl,
            status: 'ACTIVE'
        }])
        .select()
        .single();

    if (iError) return res.status(500).json({ error: iError.message });

    // 5. Record Print Audit
    await supabase.from('print_audits').insert([{
        issuance_id: issuance.id,
        printed_by: issued_by,
        printer_id: printer_id || 'unknown',
        source_ip: source_ip
    }]);

    // 6. Update Letter Status to ISSUED
    await supabase.from('letters').update({ status: 'ISSUED' }).eq('id', id);

    // 7. Audit Log
    await supabase.from('audit_logs').insert([{
        actor_id: issued_by,
        action: 'ISSUE',
        entity_type: 'LETTER',
        entity_id: id,
        metadata: { issuance_id: issuance.id, channel },
        source_ip
    }]);

    res.json({ pdf: pdfBase64 });
});

app.get('/api/verify/:hash', async (req: Request, res: Response) => {
    const { hash } = req.params;

    const { data: version, error } = await supabase
        .from('letter_versions')
        .select('*, letters(*, departments(name)), approvals(approved_at, approver_id), committee_approvals(approved_at, approver_id, committee_id), issuances(id, issued_at, status)')
        .eq('content_hash', hash)
        .single();

    if (error || !version) {
        return res.status(404).json({ valid: false, message: 'Invalid or missing document hash' });
    }

    const verificationResponse = buildVerificationResponse(version);

    await supabase.from('audit_logs').insert([{
        actor_id: null,
        action: 'VERIFY',
        entity_type: 'LETTER_VERSION',
        entity_id: version.id,
        metadata: {
            hash,
            result: verificationResponse.status,
            issuance_exists: verificationResponse.document_details.issuance_exists
        },
        source_ip: req.ip || '0.0.0.0'
    }]);

    res.json(verificationResponse);
});

app.post('/api/acknowledgements', async (req: Request, res: Response) => {
    const { letter_id, job_reference, file_url, captured_by } = req.body;
    const source_ip = req.ip || '0.0.0.0';

    if (!letter_id || !job_reference || !file_url || !captured_by) {
        return res.status(400).json({ error: 'letter_id, job_reference, file_url, and captured_by are required.' });
    }

    const { data, error } = await supabase
        .from('acknowledgements')
        .insert([{ letter_id, job_reference, file_url, captured_by }])
        .select()
        .single();

    if (error) return res.status(500).json({ error: error.message });

    await supabase.from('audit_logs').insert([{
        actor_id: captured_by,
        action: 'ACKNOWLEDGE',
        entity_type: 'LETTER',
        entity_id: letter_id,
        metadata: { job_reference, file_url },
        source_ip
    }]);

    res.status(201).json(data);
});

app.get('/api/audit-logs', async (req: Request, res: Response) => {
    const { data, error } = await supabase
        .from('audit_logs')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

    if (error) return res.status(500).json({ error: error.message });
    res.json(data);
});

app.get('/api/letters', async (req: Request, res: Response) => {
    const { context } = req.query;

    let page = parseInt(req.query.page as string) || 1;
    let limit = parseInt(req.query.limit as string) || 50;

    if (page < 1) page = 1;
    if (limit < 1) limit = 1;
    if (limit > 100) limit = 100;

    const from = (page - 1) * limit;
    const to = from + limit - 1;

    const query = supabase
        .from('letters')
        .select('*, departments(name), letter_tags(tags(name))')
        .range(from, to);

    if (context) {
        query.eq('context', String(context));
    }
    const { data, error } = await query;

    if (error) return res.status(500).json({ error: error.message });
    res.json(data);
});

app.get('/health', (req: Request, res: Response) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

export { app, supabase };

if (require.main === module) {
    app.listen(port, () => {
        console.log(`API server running on http://localhost:${port}`);
    });
}
