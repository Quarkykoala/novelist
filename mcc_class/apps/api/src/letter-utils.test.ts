import { describe, expect, it } from 'vitest';
import { buildContentHash, buildVerificationResponse, generateIssuancePdf, normalizeTagIds } from './letter-utils';

describe('normalizeTagIds', () => {
    it('filters non-strings, removes blanks, de-duplicates, and sorts', () => {
        const result = normalizeTagIds(['beta', '', 'alpha', 'beta', '  ', null, 42] as unknown[]);
        expect(result).toEqual(['alpha', 'beta']);
    });

    it('returns an empty array when input is not an array', () => {
        expect(normalizeTagIds('not-an-array')).toEqual([]);
    });
});

describe('buildContentHash', () => {
    it('produces a stable sha256 hex digest for the payload', () => {
        const hash = buildContentHash({
            letterId: 'letter-123',
            versionNumber: 2,
            context: 'COMPANY',
            departmentId: 'dept-9',
            tagIds: ['alpha', 'beta'],
            content: 'Hello world'
        });

        expect(hash).toBe('24c774facebb5311a69f31d2e1017df8cb07ce498f137226a47de905b9a7e7c4');
    });

    it('changes when content changes', () => {
        const base = {
            letterId: 'letter-123',
            versionNumber: 2,
            context: 'COMPANY',
            departmentId: 'dept-9',
            tagIds: ['alpha', 'beta']
        };
        const hashA = buildContentHash({ ...base, content: 'Hello world' });
        const hashB = buildContentHash({ ...base, content: 'Hello world!' });

        expect(hashA).not.toBe(hashB);
    });
});

describe('buildVerificationResponse', () => {
    it('prefers the most recent committee approval and marks issuances', () => {
        const response = buildVerificationResponse({
            version_number: 3,
            letters: { context: 'COMPANY', status: 'APPROVED', departments: { name: 'Legal' } },
            approvals: [{ approved_at: '2025-01-01T10:00:00Z', approver_id: 'approver-1' }],
            committee_approvals: [{
                approved_at: '2025-01-02T10:00:00Z',
                approver_id: 'approver-2',
                committee_id: 'committee-1'
            }],
            issuances: [{ id: 'iss-1' }]
        });

        expect(response.valid).toBe(true);
        expect(response.status).toBe('valid');
        expect(response.document_details.approved_via).toBe('COMMITTEE');
        expect(response.document_details.committee_id).toBe('committee-1');
        expect(response.document_details.issuance_exists).toBe(true);
    });

    it('marks revoked letters as revoked', () => {
        const response = buildVerificationResponse({
            version_number: 1,
            letters: { context: 'BCBA', status: 'REVOKED', departments: { name: 'Ops' } },
            approvals: [],
            committee_approvals: [],
            issuances: []
        });

        expect(response.valid).toBe(false);
        expect(response.status).toBe('revoked');
    });

    it('records approver details when only direct approvals exist', () => {
        const response = buildVerificationResponse({
            version_number: 2,
            letters: { context: 'COMPANY', status: 'ISSUED', departments: { name: 'Finance' } },
            approvals: [{ approved_at: '2025-01-03T10:00:00Z', approver_id: 'approver-3' }],
            committee_approvals: [],
            issuances: []
        });

        expect(response.document_details.approved_via).toBe('APPROVER');
        expect(response.document_details.committee_id).toBeNull();
        expect(response.document_details.issuance_exists).toBe(false);
    });
});

describe('generateIssuancePdf', () => {
    it('returns a PDF data URI without requiring printer access', async () => {
        const pdf = await generateIssuancePdf({
            context: 'COMPANY',
            departmentName: 'Legal',
            content: 'Test content for PDF rendering.',
            contentHash: 'deadbeefcafebabe',
            verificationUrl: 'http://localhost:5173/verify/deadbeefcafebabe',
            issuedAt: new Date('2025-01-01T00:00:00Z')
        });

        expect(pdf.startsWith('data:application/pdf')).toBe(true);
        expect(pdf).toContain('base64,');
        expect(pdf.length).toBeGreaterThan(500);
    });
});
