import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { HypothesisList, type Hypothesis } from "@/components/HypothesisList";

type HistoryEntry = {
  id: string;
  topic: string;
  status: string;
  phase?: string;
  created_at?: string;
  constraints?: Record<string, any>;
  phase_history?: { phase: string; detail?: string; timestamp?: string }[];
};

type SessionSnapshot = {
  id: string;
  topic: string;
  status: string;
  phase?: string;
  constraints?: Record<string, any>;
  phase_history?: { phase: string; detail?: string; timestamp?: string }[];
  hypotheses?: Hypothesis[];
};

export function History() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [resumeMessage, setResumeMessage] = useState("");
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchHistory = async () => {
    const sessions = await api.getSessions();
    setHistory(sessions);
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSelect = async (entry: HistoryEntry) => {
    setSelected(entry);
    setResumeMessage("");
    setLoadingDetail(true);
    try {
      const detail = await api.getSessionStatus(entry.id);
      setSnapshot({
        id: detail.id,
        topic: detail.topic,
        status: detail.status,
        phase: detail.phase,
        constraints: detail.constraints,
        phase_history: detail.phase_history,
        hypotheses: detail.hypotheses,
      });
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleResume = async () => {
    if (!selected) return;
    setResumeMessage("Launching resume run...");
    try {
      const response = await api.resumeSession(selected.id);
      setResumeMessage(`Resumed as session ${response.id}. Monitor progress in the dashboard.`);
      fetchHistory();
    } catch (err: any) {
      setResumeMessage(err?.message || "Failed to resume session.");
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6 p-6 h-[calc(100vh-64px)] overflow-hidden">
      <div className="col-span-4 h-full overflow-y-auto space-y-4 pr-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Session History</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {history.map((entry) => (
              <button
                key={entry.id}
                onClick={() => handleSelect(entry)}
                className={`w-full text-left border rounded-md p-3 transition hover:border-primary ${
                  selected?.id === entry.id ? "border-primary bg-primary/5" : "border-border"
                }`}
              >
                <div className="flex items-center justify-between text-xs uppercase tracking-wider text-muted-foreground">
                  <span>#{entry.id}</span>
                  <Badge variant="outline">{(entry.phase || entry.status).toUpperCase()}</Badge>
                </div>
                <p className="mt-1 text-sm font-semibold line-clamp-2">{entry.topic}</p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  {entry.created_at ? new Date(entry.created_at).toLocaleString() : "Unknown start"}
                </p>
              </button>
            ))}
            {history.length === 0 && (
              <p className="text-sm text-muted-foreground">No sessions recorded yet.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="col-span-8 h-full overflow-y-auto space-y-4 pr-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Session Snapshot</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!selected && <p className="text-sm text-muted-foreground">Select a session to view details.</p>}
            {selected && (
              <div className="space-y-3">
                <div>
                  <p className="text-xs uppercase text-muted-foreground">Topic</p>
                  <p className="text-lg font-semibold">{selected.topic}</p>
                </div>
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <Badge variant="secondary">{(snapshot?.phase || selected.phase || selected.status).toUpperCase()}</Badge>
                  {selected.constraints?.domains?.map((d: string) => (
                    <Badge key={d} variant="outline">{d}</Badge>
                  ))}
                </div>
                <div className="flex gap-2">
                  <Button size="sm" onClick={handleResume} disabled={loadingDetail}>
                    Resume Session
                  </Button>
                  <Button size="sm" variant="secondary" onClick={fetchHistory}>
                    Refresh History
                  </Button>
                </div>
                {resumeMessage && (
                  <p className="text-xs text-muted-foreground">{resumeMessage}</p>
                )}
                {loadingDetail && <p className="text-xs text-muted-foreground">Loading details…</p>}
                {snapshot?.constraints?.timeline && (
                  <p className="text-xs text-muted-foreground">
                    Timeline: {snapshot.constraints.timeline}
                  </p>
                )}
                {snapshot?.constraints?.dataset_links?.length ? (
                  <div className="text-xs text-muted-foreground">
                    Datasets:
                    <ul className="list-disc ml-4">
                      {snapshot.constraints.dataset_links.map((link: string) => (
                        <li key={link} className="truncate">{link}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
                {snapshot?.phase_history && snapshot.phase_history.length > 0 && (
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <p className="font-semibold uppercase tracking-wider">Phase Timeline</p>
                    <ul className="space-y-1">
                      {snapshot.phase_history.map((entry, idx) => (
                        <li key={idx} className="flex items-center gap-2">
                          <Badge variant={entry.phase === "error" ? "destructive" : "outline"}>{entry.phase}</Badge>
                          <span>{entry.detail}</span>
                          {entry.timestamp && (
                            <span className="text-[10px] text-muted-foreground/70">
                              {new Date(entry.timestamp).toLocaleString()}
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {snapshot?.hypotheses && snapshot.hypotheses.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Stored Hypotheses</CardTitle>
            </CardHeader>
            <CardContent>
              <HypothesisList hypotheses={snapshot.hypotheses} />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
