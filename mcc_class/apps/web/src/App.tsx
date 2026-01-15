import { useCallback, useEffect, useMemo, useState } from 'react';
import type { FormEvent } from 'react';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Lock, FileText, CheckCircle, ShieldCheck, History, LayoutDashboard, PlusCircle, Printer, Link as LinkIcon } from "lucide-react"

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000/api';

interface Department {
  id: string;
  name: string;
  context: string;
}

interface Tag {
  id: string;
  name: string;
  context: string;
}

interface LetterTag {
  tags: {
    name: string;
  };
}

type LetterStatus = 'DRAFT' | 'SUBMITTED' | 'APPROVED' | 'REJECTED' | 'ISSUED' | 'REVOKED';

interface Letter {
  id: string;
  context: 'COMPANY' | 'BCBA';
  status: LetterStatus;
  departments?: {
    name: string;
  };
  letter_tags?: LetterTag[];
}

interface AuditLog {
  id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

interface Committee {
  id: string;
  name: string;
  context: string;
}

interface VerificationDetails {
  context?: string;
  department?: string;
  version?: number;
  status?: string;
  approved_at?: string | null;
  approved_by?: string | null;
  approved_via?: string | null;
  committee_id?: string | null;
  issuance_exists?: boolean;
}

interface VerificationData {
  status?: string;
  valid?: boolean;
  document_details?: VerificationDetails;
}

function App() {
  const [context, setContext] = useState<'COMPANY' | 'BCBA'>('COMPANY');
  const [departments, setDepartments] = useState<Department[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);
  const [selectedDept, setSelectedDept] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [content, setContent] = useState('');
  const [letters, setLetters] = useState<Letter[]>([]);
  const [loading, setLoading] = useState(false);
  const [verificationData, setVerificationData] = useState<VerificationData | null>(null);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [view, setView] = useState<'DASHBOARD' | 'AUDIT'>('DASHBOARD');
  const [committees, setCommittees] = useState<Committee[]>([]);

  useEffect(() => {
    const hash = window.location.pathname.split('/verify/')[1];
    if (hash) {
      fetch(`${API_BASE}/verify/${hash}`)
        .then(res => res.json())
        .then(data => setVerificationData(data));
    }
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/departments?context=${context}`)
      .then(res => res.json())
      .then(setDepartments);
    fetch(`${API_BASE}/tags?context=${context}`)
      .then(res => res.json())
      .then(setTags);
  }, [context]);

  const fetchLetters = useCallback(() => {
    fetch(`${API_BASE}/letters?context=${encodeURIComponent(context)}`)
      .then(res => res.json())
      .then(setLetters);
  }, [context]);

  const fetchAuditLogs = useCallback(() => {
    fetch(`${API_BASE}/audit-logs`)
      .then(res => res.json())
      .then(setAuditLogs);
  }, []);

  const fetchCommittees = useCallback(() => {
    fetch(`${API_BASE}/committees?context=${encodeURIComponent(context)}`)
      .then(res => res.json())
      .then(setCommittees);
  }, [context]);

  useEffect(() => {
    fetchLetters();
    fetchAuditLogs();
    fetchCommittees();
  }, [fetchLetters, fetchAuditLogs, fetchCommittees]);

  const verificationStatus = useMemo(() => {
    if (!verificationData) return 'invalid';
    return verificationData.status || (verificationData.valid ? 'valid' : 'invalid');
  }, [verificationData]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/letters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          context,
          department_id: selectedDept,
          tag_ids: selectedTags,
          content,
          created_by: '00000000-0000-0000-0000-000000000000'
        })
      });
      if (res.ok) {
        setContent('');
        setSelectedTags([]);
        setSelectedDept('');
        fetchLetters();
        fetchAuditLogs();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (id: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/letters/${id}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          approver_id: '00000000-0000-0000-0000-000000000000',
          comment: 'Approved via dashboard'
        })
      });
      if (res.ok) {
        fetchLetters();
        fetchAuditLogs();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCommitteeApprove = async (id: string) => {
    if (committees.length === 0) return;
    const committeeId = committees[0].id;

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/letters/${id}/committee-approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          committee_id: committeeId,
          approver_id: '00000000-0000-0000-0000-000000000000'
        })
      });
      if (res.ok) {
        fetchLetters();
        fetchAuditLogs();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleIssue = async (id: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/letters/${id}/issue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          issued_by: '00000000-0000-0000-0000-000000000000',
          channel: 'PRINT',
          printer_id: 'HP-LASERJET-400'
        })
      });
      const data = await res.json();
      if (data.pdf) {
        const win = window.open();
        if (win) {
          win.document.write(`<iframe src="${data.pdf}" frameborder="0" style="border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;" allowfullscreen></iframe>`);
        }
        fetchLetters();
        fetchAuditLogs();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = async (id: string) => {
    const job_reference = prompt('Enter Job Reference:');
    const file_url = prompt('Enter File URL:');
    if (!job_reference || !file_url) return;

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/acknowledgements`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          letter_id: id,
          job_reference,
          file_url,
          captured_by: '00000000-0000-0000-0000-000000000000'
        })
      });
      if (res.ok) {
        fetchLetters();
        fetchAuditLogs();
      }
    } finally {
      setLoading(false);
    }
  };

  if (verificationData) {
    const details = verificationData.document_details || {};
    const approvedAt = details.approved_at
      ? new Date(details.approved_at).toLocaleString()
      : 'N/A';
    const showDetails = verificationStatus === 'valid' || verificationStatus === 'revoked';

    return (
      <div className="min-h-screen bg-background p-8 flex items-center justify-center">
        <Card className="w-full max-w-2xl shadow-2xl border-primary/20 bg-card/80 backdrop-blur-xl">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold tracking-tight">Document Verification</CardTitle>
            <CardDescription>Authentication check for evidence-backed records</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className={`p-8 rounded-xl text-center border-2 ${verificationStatus === 'valid' ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : 'bg-destructive/10 border-destructive/50 text-destructive'}`}>
              <div className="flex flex-col items-center gap-4">
                {verificationStatus === 'valid' ? <CheckCircle className="w-16 h-16" /> : <ShieldCheck className="w-16 h-16 opacity-50" />}
                <h2 className="text-2xl font-black uppercase tracking-widest">
                  {verificationStatus === 'revoked' ? 'REVOKED DOCUMENT' : (verificationStatus === 'valid' ? 'AUTHENTIC DOCUMENT' : 'INVALID OR TAMPERED')}
                </h2>
              </div>
            </div>

            {showDetails && (
              <div className="grid grid-cols-2 gap-4 text-sm bg-muted/30 p-6 rounded-lg border border-border/50">
                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Context</Label>
                    <p className="font-semibold text-foreground">{details.context || 'N/A'}</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Department</Label>
                    <p className="font-semibold text-foreground">{details.department || 'N/A'}</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Approved By</Label>
                    <p className="font-semibold text-foreground">{details.approved_by || 'N/A'}</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Status</Label>
                    <Badge variant="outline" className="mt-1 font-bold">{details.status || 'N/A'}</Badge>
                  </div>
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Approved At</Label>
                    <p className="font-semibold text-foreground">{approvedAt}</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground uppercase text-[10px] tracking-wider">Audit Trail</Label>
                    <p className="font-semibold text-foreground">{details.issuance_exists ? 'Verified Record' : 'No Issuance Found'}</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button className="w-full h-12 text-md font-bold rounded-xl" onClick={() => window.location.href = '/'}>
              Return to Dashboard
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-primary/30">
      {/* Header */}
      <div className="border-b bg-card/30 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center border border-primary/20">
              <ShieldCheck className="w-6 h-6 text-primary" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">MCC Issuance System</h1>
          </div>

          <Tabs value={view} onValueChange={(v) => setView(v as 'DASHBOARD' | 'AUDIT')} className="w-auto">
            <TabsList className="bg-background/50 border border-border p-1 h-12">
              <TabsTrigger value="DASHBOARD" className="px-6 gap-2 font-semibold">
                <LayoutDashboard className="w-4 h-4" /> Dashboard
              </TabsTrigger>
              <TabsTrigger value="AUDIT" className="px-6 gap-2 font-semibold">
                <History className="w-4 h-4" /> Audit Log
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex bg-muted/50 p-1 rounded-xl border border-border">
            <Button
              variant={context === 'COMPANY' ? "default" : "ghost"}
              size="sm"
              onClick={() => setContext('COMPANY')}
              className="rounded-lg font-bold"
            >
              Company
            </Button>
            <Button
              variant={context === 'BCBA' ? "default" : "ghost"}
              size="sm"
              onClick={() => setContext('BCBA')}
              className="rounded-lg font-bold"
            >
              BCBA
            </Button>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-10">
        {view === 'AUDIT' ? (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-3xl font-black tracking-tighter">SYSTEM AUDIT TRAIL</h2>
                <p className="text-muted-foreground text-sm font-medium">Real-time immutable ledger of all system actions</p>
              </div>
              <Badge variant="outline" className="px-4 py-1 border-primary/30 bg-primary/5 text-primary animate-pulse">
                Live Monitoring
              </Badge>
            </div>

            <Card className="border-primary/10 overflow-hidden bg-card/50">
              <Table>
                <TableHeader className="bg-muted/50">
                  <TableRow>
                    <TableHead className="w-[150px] font-bold">Action</TableHead>
                    <TableHead className="font-bold">Entity</TableHead>
                    <TableHead className="font-bold">Reference ID</TableHead>
                    <TableHead className="font-bold">Timestamp</TableHead>
                    <TableHead className="text-right font-bold">Metadata</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditLogs.map(log => (
                    <TableRow key={log.id} className="hover:bg-primary/5 transition-colors border-border/20">
                      <TableCell>
                        <Badge
                          className={`font-black uppercase tracking-tighter px-2 py-0.5 ${log.action === 'CREATE' ? 'bg-sky-500/20 text-sky-400 border-sky-500/30' :
                              log.action === 'APPROVE' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
                                log.action === 'ISSUE' ? 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30' :
                                  'bg-amber-500/20 text-amber-400 border-amber-500/30'
                            }`}
                          variant="outline"
                        >
                          {log.action}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-bold text-foreground/80">{log.entity_type}</TableCell>
                      <TableCell className="font-mono text-[11px] text-muted-foreground">{log.entity_id}</TableCell>
                      <TableCell className="text-xs text-muted-foreground">{new Date(log.created_at).toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <code className="text-[10px] bg-muted/80 p-1.5 rounded border border-border/50 text-muted-foreground">
                          {JSON.stringify(log.metadata).substring(0, 30)}...
                        </code>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </div>
        ) : (
          <div className="grid grid-cols-12 gap-10">
            {/* Left Column: Form */}
            <div className="col-span-4 space-y-8">
              <section className="space-y-6">
                <div className="flex items-center gap-2 mb-2">
                  <PlusCircle className="w-5 h-5 text-primary" />
                  <h2 className="text-xl font-black uppercase tracking-widest">New Release</h2>
                </div>

                <Card className="border-primary/20 bg-card overflow-hidden shadow-xl">
                  <form onSubmit={handleSubmit}>
                    <CardHeader className="bg-primary/5 border-b border-border/50">
                      <CardTitle className="text-lg font-bold">Letter Configuration</CardTitle>
                      <CardDescription>Select department and context specific tags</CardDescription>
                    </CardHeader>
                    <CardContent className="pt-6 space-y-6">
                      <div className="space-y-2">
                        <Label className="font-bold text-xs uppercase tracking-wider text-muted-foreground">Operating Department</Label>
                        <Select value={selectedDept} onValueChange={setSelectedDept}>
                          <SelectTrigger className="h-12 bg-background border-border/50 rounded-xl focus:ring-primary">
                            <SelectValue placeholder="Select Department" />
                          </SelectTrigger>
                          <SelectContent>
                            {departments.map(d => (
                              <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-4">
                        <Label className="font-bold text-xs uppercase tracking-wider text-muted-foreground">Security Tags</Label>
                        <div className="grid grid-cols-2 gap-3">
                          {tags.map(t => (
                            <div key={t.id} className="flex items-center space-x-3 p-3 rounded-xl border border-border/50 bg-background/50 hover:bg-primary/5 transition-all cursor-pointer">
                              <Checkbox
                                id={t.id}
                                checked={selectedTags.includes(t.id)}
                                onCheckedChange={(checked) => {
                                  if (checked) setSelectedTags([...selectedTags, t.id]);
                                  else setSelectedTags(selectedTags.filter(id => id !== t.id));
                                }}
                                className="border-primary/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
                              />
                              <label htmlFor={t.id} className="text-xs font-bold leading-none cursor-pointer text-foreground/70">{t.name}</label>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label className="font-bold text-xs uppercase tracking-wider text-muted-foreground">Document Content</Label>
                        <Textarea
                          value={content}
                          onChange={e => setContent(e.target.value)}
                          placeholder="Draft terminal content here..."
                          className="min-h-[220px] bg-background border-border/50 rounded-xl resize-none focus:ring-primary p-4"
                          required
                        />
                      </div>
                    </CardContent>
                    <CardFooter className="bg-muted/30 pt-6">
                      <Button type="submit" disabled={loading} className="w-full h-12 rounded-xl font-black text-sm uppercase tracking-widest shadow-lg shadow-primary/20">
                        {loading ? 'Processing...' : 'Generate immutable Draft'}
                      </Button>
                    </CardFooter>
                  </form>
                </Card>
              </section>
            </div>

            {/* Right Column: List */}
            <div className="col-span-8 space-y-8">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-primary" />
                  <h2 className="text-xl font-black uppercase tracking-widest">Active Records</h2>
                </div>
                <Badge variant="secondary" className="font-bold">{letters.filter(l => l.status === 'ISSUED').length} Issued</Badge>
              </div>

              <div className="grid grid-cols-1 gap-4">
                {letters.map((l) => (
                  <Card
                    key={l.id}
                    className={`group transition-all duration-300 border-border/50 ${l.status === 'APPROVED' || l.status === 'ISSUED' ? 'bg-muted/20 border-primary/10' : 'bg-card hover:border-primary/30'}`}
                  >
                    <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                      <div className="space-y-1">
                        <CardTitle className="text-lg font-bold flex items-center gap-3">
                          {l.departments?.name}
                          {(l.status === 'APPROVED' || l.status === 'ISSUED') && <Lock className="w-4 h-4 text-emerald-400" />}
                        </CardTitle>
                        <div className="flex gap-2">
                          <Badge variant="outline" className={`font-bold text-[10px] ${l.context === 'COMPANY' ? 'text-sky-400 border-sky-400/20 bg-sky-400/5' : 'text-amber-400 border-amber-400/20 bg-amber-400/5'}`}>
                            {l.context}
                          </Badge>
                          <Badge className="font-black text-[10px] uppercase tracking-tighter" variant={l.status === 'DRAFT' ? 'secondary' : 'default'}>
                            {l.status}
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {l.letter_tags?.map((lt) => (
                          <Badge key={lt.tags.name} variant="outline" className="text-[10px] bg-muted/50 border-border/50 text-muted-foreground px-3">
                            {lt.tags.name}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                    <CardFooter className="flex justify-end gap-3 pt-4 border-t border-border/20 mt-4 bg-muted/10 group-hover:bg-muted/20 transition-colors">
                      {l.status === 'DRAFT' && (
                        <>
                          {l.context === 'COMPANY' ? (
                            <Button variant="default" className="gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-bold" onClick={() => handleApprove(l.id)} disabled={loading}>
                              <CheckCircle className="w-3.5 h-3.5" /> Approve
                            </Button>
                          ) : (
                            <Button variant="default" className="gap-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-xs font-bold" onClick={() => handleCommitteeApprove(l.id)} disabled={loading}>
                              <ShieldCheck className="w-3.5 h-3.5" /> Committee Approve
                            </Button>
                          )}
                        </>
                      )}
                      {l.status === 'APPROVED' && (
                        <Button variant="default" className="gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-xs font-bold" onClick={() => handleIssue(l.id)} disabled={loading}>
                          <Printer className="w-3.5 h-3.5" /> Issue Professional PDF
                        </Button>
                      )}
                      {l.status === 'ISSUED' && (
                        <Button variant="outline" className="gap-2 rounded-lg border-primary/30 text-primary hover:bg-primary/5 text-xs font-bold" onClick={() => handleAcknowledge(l.id)} disabled={loading}>
                          <LinkIcon className="w-3.5 h-3.5" /> Link Acknowledgement
                        </Button>
                      )}
                    </CardFooter>
                  </Card>
                ))}
                {letters.length === 0 && (
                  <div className="py-20 text-center border-2 border-dashed border-border rounded-3xl opacity-40">
                    <p className="text-xl font-bold tracking-tight">No letters identified in current sector</p>
                    <p className="text-sm font-medium">Initialize a new release from the left terminal</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
