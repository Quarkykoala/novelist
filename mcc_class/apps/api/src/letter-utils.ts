import { jsPDF } from 'jspdf';
import QRCode from 'qrcode';
import crypto from 'crypto';

export const normalizeTagIds = (tagIds: unknown): string[] => {
    if (!Array.isArray(tagIds)) return [];
    const cleaned = tagIds.filter((id): id is string => typeof id === 'string' && id.trim().length > 0);
    return Array.from(new Set(cleaned)).sort();
};

export const buildContentHash = ({
    letterId,
    versionNumber,
    context,
    departmentId,
    tagIds,
    content
}: {
    letterId: string;
    versionNumber: number;
    context: string;
    departmentId: string;
    tagIds: string[];
    content: string;
}): string => {
    const payload = {
        letter_id: letterId,
        version: versionNumber,
        context,
        department_id: departmentId,
        tag_ids: tagIds,
        content
    };
    return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
};

type GeneratePdfInput = {
    context: string;
    departmentName?: string;
    content: string;
    contentHash: string;
    verificationUrl: string;
    issuedAt?: Date;
};

export const generateIssuancePdf = async ({
    context,
    departmentName,
    content,
    contentHash,
    verificationUrl,
    issuedAt
}: GeneratePdfInput): Promise<string> => {
    const doc = new jsPDF();

    doc.setFontSize(22);
    doc.setTextColor(79, 70, 229);
    doc.text(context === 'COMPANY' ? 'MCC COMPANY OPS' : 'BCBA ASSOCIATION', 20, 30);

    doc.setFontSize(10);
    doc.setTextColor(150);
    doc.text(`Department: ${departmentName}`, 20, 40);
    doc.text(`Date: ${(issuedAt ?? new Date()).toLocaleDateString()}`, 20, 45);

    doc.setDrawColor(200);
    doc.line(20, 50, 190, 50);

    doc.setFontSize(12);
    doc.setTextColor(0);
    const splitText = doc.splitTextToSize(content, 170);
    doc.text(splitText, 20, 70);

    const qrDataUrl = await QRCode.toDataURL(verificationUrl);
    doc.addImage(qrDataUrl, 'PNG', 150, 240, 40, 40);
    doc.setFontSize(8);
    doc.text('Scan to Verify Authenticity', 153, 282);
    doc.text(`Hash: ${contentHash.substring(0, 16)}...`, 20, 282);

    return doc.output('datauristring');
};

type ApprovalRecord = {
    approved_at?: string | null;
    approver_id?: string | null;
    committee_id?: string | null;
};

type VerificationVersion = {
    version_number: number;
    letters?: {
        context?: string | null;
        status?: string | null;
        departments?: { name?: string | null } | null;
    } | null;
    approvals?: ApprovalRecord[] | null;
    committee_approvals?: ApprovalRecord[] | null;
    issuances?: { id?: string }[] | null;
};

type VerificationResponse = {
    valid: boolean;
    status: 'valid' | 'invalid' | 'revoked';
    document_details: {
        context?: string | null;
        department?: string | null;
        version: number;
        status?: string | null;
        approved_at: string | null;
        approved_by: string | null;
        approved_via: 'COMMITTEE' | 'APPROVER' | null;
        committee_id: string | null;
        issuance_exists: boolean;
    };
};

export const buildVerificationResponse = (version: VerificationVersion): VerificationResponse => {
    const letterStatus = version.letters?.status;
    const issuanceExists = (version.issuances || []).length > 0;
    let validity: 'valid' | 'invalid' | 'revoked' = 'invalid';
    if (letterStatus === 'REVOKED') {
        validity = 'revoked';
    } else if (letterStatus === 'APPROVED' || letterStatus === 'ISSUED') {
        validity = 'valid';
    }

    const approvalCandidates = [
        ...(version.approvals || []),
        ...(version.committee_approvals || [])
    ];
    const approval = approvalCandidates.sort((a, b) => {
        const aTime = a.approved_at ? Date.parse(a.approved_at) : 0;
        const bTime = b.approved_at ? Date.parse(b.approved_at) : 0;
        return bTime - aTime;
    })[0];
    const approvedVia = approval && 'committee_id' in approval ? 'COMMITTEE' : approval ? 'APPROVER' : null;
    const committeeId = approval && 'committee_id' in approval ? approval.committee_id ?? null : null;

    return {
        valid: validity === 'valid',
        status: validity,
        document_details: {
            context: version.letters?.context,
            department: version.letters?.departments?.name,
            version: version.version_number,
            status: letterStatus,
            approved_at: approval?.approved_at || null,
            approved_by: approval?.approver_id || null,
            approved_via: approvedVia,
            committee_id: committeeId,
            issuance_exists: issuanceExists
        }
    };
};
