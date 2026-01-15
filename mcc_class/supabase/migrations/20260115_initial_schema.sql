-- Initial Schema for Evidence-Backed Letter Issuance System

-- Enums for context and status
CREATE TYPE app_context AS ENUM ('COMPANY', 'BCBA');
CREATE TYPE letter_status AS ENUM ('DRAFT', 'SUBMITTED', 'APPROVED', 'REJECTED', 'ISSUED', 'REVOKED');
CREATE TYPE issuance_channel AS ENUM ('PRINT', 'EMAIL', 'COURIER');

-- 1. Departments
CREATE TABLE departments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    context app_context NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Master Tags
CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    context app_context NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Letters (Base Record)
CREATE TABLE letters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context app_context NOT NULL,
    department_id UUID REFERENCES departments(id) NOT NULL,
    status letter_status DEFAULT 'DRAFT' NOT NULL,
    created_by UUID REFERENCES auth.users(id) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 4. Letter Tags (Many-to-Many)
CREATE TABLE letter_tags (
    letter_id UUID REFERENCES letters(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (letter_id, tag_id)
);

-- 5. Letter Versions (Immutable snapshots)
CREATE TABLE letter_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    letter_id UUID REFERENCES letters(id) NOT NULL,
    version_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- SHA-256 of content
    created_by UUID REFERENCES auth.users(id) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(letter_id, version_number)
);

-- 6. Approvals
CREATE TABLE approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    letter_version_id UUID REFERENCES letter_versions(id) NOT NULL,
    approver_id UUID REFERENCES auth.users(id) NOT NULL,
    comment TEXT,
    approved_at TIMESTAMPTZ DEFAULT now()
);

-- 7. Issuance Records
CREATE TABLE issuances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    letter_version_id UUID REFERENCES letter_versions(id) NOT NULL,
    issued_by UUID REFERENCES auth.users(id) NOT NULL,
    issued_at TIMESTAMPTZ DEFAULT now(),
    channel issuance_channel NOT NULL,
    qr_payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

-- 8. Print Audit Trail
CREATE TABLE print_audits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issuance_id UUID REFERENCES issuances(id) NOT NULL,
    printed_by UUID REFERENCES auth.users(id) NOT NULL,
    printed_at TIMESTAMPTZ DEFAULT now(),
    printer_id TEXT DEFAULT 'unknown',
    source_ip INET NOT NULL
);

-- 9. Acknowledgements
CREATE TABLE acknowledgements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    letter_id UUID REFERENCES letters(id) NOT NULL,
    job_reference TEXT,
    file_url TEXT NOT NULL,
    captured_by UUID REFERENCES auth.users(id) NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT now()
);

-- 10. Committees (BCBA Governance)
CREATE TABLE committees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    context app_context NOT NULL DEFAULT 'BCBA',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 11. Committee Approvals
CREATE TABLE committee_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    committee_id UUID REFERENCES committees(id) NOT NULL,
    letter_version_id UUID REFERENCES letter_versions(id) NOT NULL,
    approver_id UUID REFERENCES auth.users(id) NOT NULL,
    approved_at TIMESTAMPTZ DEFAULT now()
);

-- 12. Audit Log (Append-Only)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id UUID REFERENCES auth.users(id),
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id UUID NOT NULL,
    metadata JSONB,
    source_ip INET,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- RLS Policies (Basic Internal-Only Access)
ALTER TABLE departments ENABLE ROW LEVEL SECURITY;
ALTER TABLE tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE letters ENABLE ROW LEVEL SECURITY;
ALTER TABLE letter_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE issuances ENABLE ROW LEVEL SECURITY;
ALTER TABLE print_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE acknowledgements ENABLE ROW LEVEL SECURITY;
ALTER TABLE committees ENABLE ROW LEVEL SECURITY;
ALTER TABLE committee_approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Default policy: Auth users can read everything (internal system)
CREATE POLICY "Internal Read Access" ON departments FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON tags FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON letters FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON letter_versions FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON approvals FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON issuances FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON print_audits FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON acknowledgements FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON committees FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON committee_approvals FOR SELECT TO authenticated USING (true);
CREATE POLICY "Internal Read Access" ON audit_logs FOR SELECT TO authenticated USING (true);
