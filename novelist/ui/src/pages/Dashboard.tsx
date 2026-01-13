import { useState, useEffect, useRef } from "react";
import { api, type SessionConstraintsPayload } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Play, RotateCcw, Activity, Send, Lock, Unlock, RefreshCcw } from "lucide-react";
import { Reactor } from "@/components/Reactor";
import { SoulFeed, type SoulMessage } from "@/components/SoulFeed";
import { HypothesisList, type Hypothesis } from "@/components/HypothesisList";

type PhaseRecord = {
  phase: string;
  detail?: string;
  timestamp?: string;
};

export function Dashboard() {
  const [topic, setTopic] = useState("The effects of intermittent fasting on aging");
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState("Idle");
  const [statusDetail, setStatusDetail] = useState("");
  const [phase, setPhase] = useState("queued");
  const [phaseHistory, setPhaseHistory] = useState<PhaseRecord[]>([]);
  const [loop, setLoop] = useState(0);
  const [maxLoops] = useState(4);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [gaps, setGaps] = useState<any[]>([]);
  const [sourceMetadata, setSourceMetadata] = useState<Record<string, any>>({});
  const [soulMessages, setSoulMessages] = useState<SoulMessage[]>([]);
  const [chatMessage, setChatMessage] = useState("");
  const [logs, setLogs] = useState<{label: string, text: string, ts: Date}[]>([]);
  const [domainsInput, setDomainsInput] = useState("");
  const [modalitiesInput, setModalitiesInput] = useState("");
  const [timelineInput, setTimelineInput] = useState("");
  const [datasetInput, setDatasetInput] = useState("");
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [activeConstraints, setActiveConstraints] = useState<SessionConstraintsPayload | null>(null);
  const [personas, setPersonas] = useState<any[]>([]);
  
  const pollInterval = useRef<any>(null);

  const addLog = (label: string, text: string) => {
    setLogs(prev => [{label, text, ts: new Date()}, ...prev].slice(0, 20));
  };

  const handleSendMessage = async () => {
    if (!chatMessage || !sessionId) return;
    const msg = chatMessage;
    setChatMessage("");
    addLog("user", msg);
    
    try {
      await api.sendChatMessage(sessionId, msg);
    } catch (err: any) {
      addLog("error", "Failed to send: " + err.message);
    }
  };

  const handleLockPersona = async (personaId: string, currentLocked: boolean) => {
    if (!sessionId) return;
    try {
      if (currentLocked) {
        await api.unlockPersona(sessionId, personaId);
        addLog("system", `Unlocked persona: ${personaId}`);
      } else {
        await api.lockPersona(sessionId, personaId);
        addLog("system", `Locked persona: ${personaId}`);
      }
      // Status poll will pick up the change, but we can update locally for snappiness
      setPersonas(prev => prev.map(p => p.id === personaId ? { ...p, locked: !currentLocked } : p));
    } catch (err: any) {
      addLog("error", "Failed to toggle lock: " + err.message);
    }
  };

  const handleRegeneratePersona = async (personaId: string) => {
    if (!sessionId) return;
    try {
      await api.regeneratePersona(sessionId, personaId);
      addLog("system", `Regenerating persona: ${personaId}`);
    } catch (err: any) {
      addLog("error", "Failed to regenerate: " + err.message);
    }
  };

  const handleWeightChange = async (personaId: string, weight: number) => {
    if (!sessionId) return;
    try {
      await api.updatePersonaWeight(sessionId, personaId, weight);
      setPersonas(prev => prev.map(p => p.id === personaId ? { ...p, weight } : p));
    } catch (err: any) {
      addLog("error", "Failed to update weight: " + err.message);
    }
  };

  const parseList = (value: string) => value.split(",").map((entry) => entry.trim()).filter(Boolean);
  const parseDatasetLinks = (value: string) => value.split(/[\n,]+/).map((entry) => entry.trim()).filter(Boolean);
  const isValidUrl = (value: string) => {
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  };

  const handleGenerate = async () => {
    if (!topic) return;
    
    setIsGenerating(true);
    setHypotheses([]);
    setSoulMessages([]);
    setLogs([]);
    setLoop(0);
    setStatus("Queued");
    setStatusDetail("Queued for execution");
    setPhase("queued");
    setPhaseHistory([{ phase: "queued", detail: "Queued for execution", timestamp: new Date().toISOString() }]);
    setPersonas([]);

    const domains = parseList(domainsInput);
    const modalities = parseList(modalitiesInput);
    const datasetLinks = parseDatasetLinks(datasetInput);
    const invalidLink = datasetLinks.find((link) => !isValidUrl(link));
    if (invalidLink) {
      setDatasetError(`Invalid dataset URL: ${invalidLink}`);
      setIsGenerating(false);
      return;
    }
    setDatasetError(null);

    const constraintsPayload: SessionConstraintsPayload = {
      domains,
      modalities,
      timeline: timelineInput || undefined,
      dataset_links: datasetLinks,
    };

    setActiveConstraints(constraintsPayload);

    try {
      const session = await api.startSession(topic, {
        maxIterations: maxLoops,
        constraints: constraintsPayload,
      });
      setSessionId(session.id);
      setPhase(session.phase || "queued");
      setStatus((session.phase || "queued").replace(/^./, (c: string) => c.toUpperCase()));
      setStatusDetail("Queued for execution");
      addLog("system", "Session started: " + session.id);
    } catch (err: any) {
      console.error(err);
      setStatus("Error: " + err.message);
      setIsGenerating(false);
      addLog("error", err.message);
    }
  };

  useEffect(() => {
    if (sessionId && isGenerating) {
      pollInterval.current = setInterval(async () => {
        try {
          const s = await api.getSessionStatus(sessionId);
          
          const phaseValue = s.phase || "processing";
          const prettyPhase = phaseValue.charAt(0).toUpperCase() + phaseValue.slice(1);
          setStatus(prettyPhase);
          setPhase(phaseValue);
          setLoop(s.iteration || 0);
          if (s.phase_history) setPhaseHistory(s.phase_history);
          setStatusDetail(s.status_detail || "");
          if (s.constraints) setActiveConstraints(s.constraints);
          if (s.personas) setPersonas(s.personas);
          
          if (s.hypotheses) setHypotheses(s.hypotheses);
          if (s.gaps) setGaps(s.gaps);
          if (s.source_metadata) setSourceMetadata(s.source_metadata);
          if (s.soulMessages) setSoulMessages(s.soulMessages); 
          
          if (s.complete) {
            setIsGenerating(false);
            setStatus("Complete");
            setStatusDetail(s.status_detail || "Session finalized");
            if (pollInterval.current) clearInterval(pollInterval.current);
            addLog("system", "Generation complete");
          }
          
          if (s.error) {
             setIsGenerating(false);
             setStatus("Error");
             setStatusDetail(s.error);
             addLog("error", s.error);
             if (pollInterval.current) clearInterval(pollInterval.current);
          }

        } catch (err) {
          console.error("Poll error", err);
        }
      }, 2000);
    }

    return () => {
      if (pollInterval.current) clearInterval(pollInterval.current);
    };
  }, [sessionId, isGenerating]);

  return (
    <div className="grid grid-cols-12 gap-6 p-6 h-[calc(100vh-64px)] overflow-hidden">
      {/* LEFT COLUMN: Input & Status */}
      <div className="col-span-3 flex flex-col gap-6 h-full overflow-y-auto pr-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              Research Input
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
                <label className="text-sm font-medium">Research Topic</label>
                <Input 
                  value={topic} 
                  onChange={(e) => setTopic(e.target.value)} 
                  placeholder="e.g. CRISPR delivery..."
                />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Domains</label>
              <Input 
                value={domainsInput}
                onChange={(e) => setDomainsInput(e.target.value)}
                placeholder="comma separated e.g. longevity, systems biology"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Modalities</label>
              <Input 
                value={modalitiesInput}
                onChange={(e) => setModalitiesInput(e.target.value)}
                placeholder="e.g. simulation, wet lab"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Timeline Pressure</label>
              <Input 
                value={timelineInput}
                onChange={(e) => setTimelineInput(e.target.value)}
                placeholder="e.g. deliver recommendations within 2 weeks"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Dataset Links</label>
              <textarea
                value={datasetInput}
                onChange={(e) => setDatasetInput(e.target.value)}
                placeholder="One URL per line or comma separated"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                rows={3}
              />
              {datasetError && (
                <p className="text-xs text-destructive">{datasetError}</p>
              )}
            </div>
            
            <div className="space-y-2">
                <label className="text-sm font-medium">Sources</label>
                <div className="flex gap-2">
                    <Badge variant="secondary" className="cursor-pointer hover:bg-secondary/80">arXiv</Badge>
                    <Badge variant="outline" className="opacity-50 cursor-not-allowed">PubMed</Badge>
                </div>
            </div>

            {activeConstraints && (
              <div className="rounded-md border bg-muted/30 p-3 space-y-1 text-xs">
                <p className="font-semibold uppercase text-[10px] tracking-wider text-muted-foreground">Active constraints</p>
                <div className="flex flex-wrap gap-2">
                  {activeConstraints.domains?.map((d) => (
                    <Badge key={d} variant="outline">{d}</Badge>
                  ))}
                  {activeConstraints.modalities?.map((m) => (
                    <Badge key={m} variant="secondary">{m}</Badge>
                  ))}
                </div>
                {activeConstraints.timeline && (
                  <p className="text-muted-foreground">Timeline: {activeConstraints.timeline}</p>
                )}
                {activeConstraints.dataset_links?.length ? (
                  <div className="text-muted-foreground/80">
                    Datasets:
                    <ul className="list-disc ml-4">
                      {activeConstraints.dataset_links.map((link) => (
                        <li key={link} className="truncate">{link}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            )}

            {personas.length > 0 && (
              <div className="rounded-md border bg-muted/30 p-3 space-y-2 text-xs">
                <p className="font-semibold uppercase text-[10px] tracking-wider text-muted-foreground">Persona roster</p>
                <div className="space-y-2">
                  {personas.map((persona) => (
                    <div key={`${persona.soul_role}-${persona.name}`} className="rounded-md border bg-background/80 p-2 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{persona.name}</span>
                        <Badge variant="outline" className="text-[10px] uppercase">
                          {persona.soul_role?.replace("_", " ")}
                        </Badge>
                      </div>
                      <p className="text-muted-foreground">{persona.role}</p>
                      {persona.objective && (
                        <p className="text-muted-foreground/80 italic line-clamp-2">"{persona.objective}"</p>
                      )}
                      
                      <div className="flex flex-col gap-1 pt-1 border-t border-muted">
                        <div className="flex items-center justify-between">
                          <label className="text-[10px] text-muted-foreground">Influence Weight</label>
                          <span className="text-[10px] font-mono">{Math.round((persona.weight || 0) * 100)}%</span>
                        </div>
                        <input 
                          type="range" 
                          min="0" 
                          max="1" 
                          step="0.05"
                          value={persona.weight || 0}
                          onChange={(e) => handleWeightChange(persona.id, parseFloat(e.target.value))}
                          className="w-full h-1 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                      </div>

                      <div className="flex gap-2 pt-1">
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7 px-2 text-[10px] gap-1"
                          onClick={() => handleLockPersona(persona.id, !!persona.locked)}
                        >
                          {persona.locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                          {persona.locked ? "Locked" : "Lock"}
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7 px-2 text-[10px] gap-1"
                          onClick={() => handleRegeneratePersona(persona.id)}
                          disabled={persona.locked}
                        >
                          <RefreshCcw className="w-3 h-3" />
                          Regenerate
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <Button 
                className="w-full mt-4" 
                size="lg" 
                onClick={handleGenerate} 
                disabled={isGenerating || !!datasetError}
            >
              {isGenerating ? (
                  <>
                    <RotateCcw className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
              ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Research
                  </>
              )}
            </Button>
          </CardContent>
        </Card>

        <Card className="flex-1 min-h-[200px] flex flex-col">
            <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wider text-muted-foreground">
                    <Activity className="w-4 h-4" />
                    Activity Log
                </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto space-y-3">
                {logs.length === 0 && (
                    <div className="text-xs text-muted-foreground text-center py-4">Waiting to start...</div>
                )}
                {logs.map((log, i) => (
                    <div key={i} className="flex gap-3 text-xs">
                        <span className="text-muted-foreground font-mono whitespace-nowrap">
                            {log.ts.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})}
                        </span>
                        <div>
                            <span className="font-semibold mr-2 uppercase text-[10px] bg-muted px-1 rounded">{log.label}</span>
                            <span className="text-foreground/80">{log.text}</span>
                        </div>
                    </div>
                ))}
            </CardContent>
        </Card>
      </div>

      {/* MIDDLE COLUMN: Reactor & Results */}
      <div className="col-span-6 flex flex-col gap-6 h-full overflow-y-auto pr-2 pb-20">
        <Card>
            <CardContent className="pt-6">
                <Reactor 
                  isActive={isGenerating} 
                  status={status} 
                  phase={phase}
                  history={phaseHistory}
                  statusDetail={statusDetail}
                />
                
                <div className="flex justify-center mt-4 gap-8 text-sm text-muted-foreground">
                    <div className="flex flex-col items-center">
                        <span className="font-bold text-2xl text-foreground">{loop} / {maxLoops}</span>
                        <span>Iterations</span>
                    </div>
                    <div className="w-px bg-border" />
                    <div className="flex flex-col items-center">
                        <span className="font-bold text-2xl text-foreground">{hypotheses.length}</span>
                        <span>Hypotheses</span>
                    </div>
                </div>
            </CardContent>
        </Card>

        {gaps.length > 0 && (
            <div className="space-y-4">
                <h2 className="text-xl font-bold tracking-tight flex items-center gap-2">
                    <Activity className="w-5 h-5 text-primary" />
                    Knowledge Gaps Discovered
                </h2>
                <div className="grid grid-cols-2 gap-4">
                    {gaps.map((gap, i) => (
                        <Card key={i} className="border-primary/20 bg-primary/5">
                            <CardContent className="p-4 space-y-2">
                                <div className="flex items-center justify-between">
                                    <Badge variant="outline" className="text-[10px] uppercase">{gap.type || 'Missing Connection'}</Badge>
                                    <span className="text-[10px] text-muted-foreground font-mono">ID: GAP-{i+1}</span>
                                </div>
                                <p className="text-sm font-semibold">{gap.concept_a} × {gap.concept_b}</p>
                                <p className="text-xs text-muted-foreground line-clamp-2">{gap.description}</p>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>
        )}

        <div className="space-y-4">
            <h2 className="text-xl font-bold tracking-tight">Generated Hypotheses</h2>
            <HypothesisList hypotheses={hypotheses} sourceMetadata={sourceMetadata} />
        </div>

        {isGenerating && (
            <Card className="border-primary/50 shadow-lg shadow-primary/10 mt-8">
                <CardHeader className="py-3 px-4 border-b bg-primary/5">
                    <CardTitle className="text-xs uppercase tracking-widest flex items-center gap-2">
                        <Activity className="w-3 h-3 text-primary" />
                        Research Assistant
                    </CardTitle>
                </CardHeader>
                <CardContent className="p-3">
                    <div className="flex gap-2">
                        <Input 
                            value={chatMessage} 
                            onChange={(e) => setChatMessage(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                            placeholder="Direct the collective (e.g. 'Focus on cost', 'Why ignore X?')..."
                            className="flex-1"
                        />
                        <Button size="icon" onClick={handleSendMessage} disabled={!chatMessage}>
                            <Send className="w-4 h-4" />
                        </Button>
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2 px-1 italic">
                        Your guidance will be integrated into the next iteration.
                    </p>
                </CardContent>
            </Card>
        )}
      </div>

      {/* RIGHT COLUMN: Soul Feed */}
      <div className="col-span-3 h-full overflow-hidden flex flex-col">
        <h3 className="font-semibold mb-4 px-1">Collective Intelligence</h3>
        <SoulFeed messages={soulMessages} />
      </div>
    </div>
  );
}
