import { useState, useEffect, useRef } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Play, RotateCcw, Activity, Send } from "lucide-react";
import { Reactor } from "@/components/Reactor";
import { SoulFeed, type SoulMessage } from "@/components/SoulFeed";
import { HypothesisList, type Hypothesis } from "@/components/HypothesisList";

export function Dashboard() {
  const [topic, setTopic] = useState("The effects of intermittent fasting on aging");
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState("Idle");
  const [loop, setLoop] = useState(0);
  const [maxLoops] = useState(4);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [gaps, setGaps] = useState<any[]>([]);
  const [sourceMetadata, setSourceMetadata] = useState<Record<string, any>>({});
  const [soulMessages, setSoulMessages] = useState<SoulMessage[]>([]);
  const [chatMessage, setChatMessage] = useState("");
  const [logs, setLogs] = useState<{label: string, text: string, ts: Date}[]>([]);
  
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

  const handleGenerate = async () => {
    if (!topic) return;
    
    setIsGenerating(true);
    setHypotheses([]);
    setSoulMessages([]);
    setLogs([]);
    setLoop(0);
    setStatus("Initializing...");

    try {
      const session = await api.startSession(topic);
      setSessionId(session.id);
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
          
          setStatus(s.phase || "Processing");
          setLoop(s.iteration || 0);
          
          if (s.hypotheses) setHypotheses(s.hypotheses);
          if (s.gaps) setGaps(s.gaps);
          if (s.source_metadata) setSourceMetadata(s.source_metadata);
          if (s.soulMessages) setSoulMessages(s.soulMessages); 
          
          if (s.complete) {
            setIsGenerating(false);
            setStatus("Complete");
            if (pollInterval.current) clearInterval(pollInterval.current);
            addLog("system", "Generation complete");
          }
          
          if (s.error) {
             setIsGenerating(false);
             setStatus("Error");
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
                <label className="text-sm font-medium">Sources</label>
                <div className="flex gap-2">
                    <Badge variant="secondary" className="cursor-pointer hover:bg-secondary/80">arXiv</Badge>
                    <Badge variant="outline" className="opacity-50 cursor-not-allowed">PubMed</Badge>
                </div>
            </div>

            <Button 
                className="w-full mt-4" 
                size="lg" 
                onClick={handleGenerate} 
                disabled={isGenerating}
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
                <Reactor isActive={isGenerating} status={status} />
                
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
