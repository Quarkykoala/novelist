import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  CheckCircle2, 
  XCircle, 
  Link as LinkIcon, 
  FileText, 
  ExternalLink, 
  Image as ImageIcon,
  ThumbsUp,
  ThumbsDown,
  Search,
  RotateCcw
} from "lucide-react";

export interface Hypothesis {
  id: string;
  statement: string;
  rationale?: string;
  supporting_papers?: string[];
  diagram?: string;
  simulation_result?: {
    success: boolean;
    supports_hypothesis: boolean;
    code: string;
    plot_path?: string;
    vision_commentary?: string;
    status?: string;
    timestamp?: string;
  };
  simulation_history?: any[];
}

interface HypothesisListProps {
  hypotheses: Hypothesis[];
  sourceMetadata?: Record<string, any>;
  onVote?: (id: string, direction: "up" | "down") => void;
  onInvestigate?: (id: string) => void;
  onRerun?: (id: string) => void;
}

function getPlotFilename(path: string): string {
  try {
    return path.split(/[/\\]/).pop() || "";
  } catch (e) {
    return "";
  }
}

export function HypothesisList({ hypotheses, sourceMetadata = {}, onVote, onInvestigate, onRerun }: HypothesisListProps) {
  return (
    <div className="space-y-4">
      {hypotheses.map((h, i) => {
        const hasMechanism = h.rationale?.includes("→");
        const simResult = h.simulation_result;
        const isVerified = simResult?.success && simResult?.supports_hypothesis;

        return (
          <Card key={i} className="overflow-hidden animate-in fade-in slide-in-from-bottom-4 group" style={{ animationDelay: `${i * 100}ms` }}>
            <CardContent className="p-4 flex gap-4">
              <div className="flex flex-col items-center gap-2">
                <div className="text-4xl font-black text-muted/20 font-mono select-none">
                    {i + 1}
                </div>
                {onVote && (
                    <div className="flex flex-col gap-1">
                        <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-8 w-8 text-muted-foreground hover:text-primary"
                            onClick={() => onVote(h.id, "up")}
                        >
                            <ThumbsUp className="w-4 h-4" />
                        </Button>
                        <Button 
                            variant="ghost" 
                            size="icon" 
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            onClick={() => onVote(h.id, "down")}
                        >
                            <ThumbsDown className="w-4 h-4" />
                        </Button>
                    </div>
                )}
              </div>
              
              <div className="flex-1 space-y-2">
                <div className="flex items-start justify-between gap-4">
                    <div className="space-y-1">
                        <p className="font-medium text-lg leading-snug">
                        {h.statement}
                        </p>
                        <span className="text-[10px] text-muted-foreground font-mono uppercase">ID: {h.id}</span>
                    </div>
                    <div className="flex gap-2 shrink-0">
                        {onInvestigate && (
                            <Button 
                                variant="outline" 
                                size="sm" 
                                className="h-8 gap-2 text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                                onClick={() => onInvestigate(h.id)}
                            >
                                <Search className="w-3 h-3" />
                                Investigate
                            </Button>
                        )}
                    </div>
                </div>

                {/* Tags and Metadata */}
                <div className="flex flex-wrap gap-2 items-center">
                    {hasMechanism && (
                    <div className="inline-flex items-center gap-2 px-2 py-1 bg-violet-500/10 text-violet-600 dark:text-violet-300 rounded text-xs font-medium">
                        <LinkIcon className="w-3 h-3" />
                        Mechanism: {h.rationale?.split('.')[0].slice(0, 50)}...
                    </div>
                    )}
                    
                    {h.supporting_papers && h.supporting_papers.length > 0 && (
                        h.supporting_papers.map((pid, idx) => {
                            const paper = sourceMetadata[pid];
                            return (
                                <a 
                                    key={idx} 
                                    href={paper?.abs_url || `https://arxiv.org/abs/${pid}`}
                                    target="_blank" 
                                    rel="noreferrer"
                                    title={paper?.title || pid}
                                    className="inline-flex items-center gap-1 px-2 py-1 bg-blue-500/10 text-blue-600 hover:bg-blue-500/20 rounded text-[10px] transition-colors max-w-[200px]"
                                >
                                    <ExternalLink className="w-3 h-3 flex-shrink-0" />
                                    <span className="truncate">{paper?.title || pid}</span>
                                </a>
                            );
                        })
                    )}
                </div>

                {/* Visual Mechanism (SVG) */}
                {h.diagram && (
                    <div className="mt-4 rounded-md border bg-white p-4 flex flex-col gap-2 shadow-inner">
                        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground tracking-wider">
                            <ImageIcon className="w-3 h-3" />
                            Mechanism Visualization
                        </div>
                        <div 
                            className="w-full overflow-hidden flex justify-center"
                            dangerouslySetInnerHTML={{ __html: h.diagram }}
                        />
                    </div>
                )}

                {/* Simulation Result */}
                {simResult && (
                  <div className="mt-4 rounded-md border bg-muted/50 overflow-hidden shadow-sm">
                    <div className="p-2 border-b bg-muted/30 flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground tracking-wider">
                            <FileText className="w-3 h-3" />
                            Simulation Lab
                        </div>
                        <div className="flex items-center gap-3">
                            {onRerun && (
                                <Button 
                                    variant="ghost" 
                                    size="sm" 
                                    className="h-6 text-[10px] gap-1 px-2 hover:bg-primary/10 hover:text-primary"
                                    onClick={() => onRerun(h.id)}
                                >
                                    <RotateCcw className="w-3 h-3" />
                                    Rerun
                                </Button>
                            )}
                            <Badge variant={isVerified ? "default" : "destructive"} className="gap-1 text-[10px]">
                                {isVerified ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                                {isVerified ? "Verified In-Silico" : "Verification Failed"}
                            </Badge>
                        </div>
                    </div>
                    
                    <div className="p-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                            <div className="space-y-1">
                                <p className="text-[10px] font-bold uppercase text-muted-foreground">Model Logic</p>
                                <pre className="text-[10px] font-mono bg-background p-2 rounded border overflow-x-auto max-h-48">
                                    {simResult.code}
                                </pre>
                            </div>
                            
                            {simResult.vision_commentary && (
                                <div className="space-y-1">
                                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Gemini Vision Analysis</p>
                                    <div className="text-[11px] bg-primary/5 p-2 rounded border border-primary/10 italic text-foreground/90">
                                        {simResult.vision_commentary}
                                    </div>
                                </div>
                            )}
                        </div>
                        
                        <div className="space-y-2">
                            {simResult.plot_path && (
                                <div className="rounded overflow-hidden border bg-white flex flex-col">
                                    <img 
                                        src={`/plots/${getPlotFilename(simResult.plot_path)}`}
                                        alt="Simulation Plot" 
                                        className="w-full h-auto object-contain"
                                    />
                                    <div className="p-1.5 border-t bg-muted/20 text-center">
                                        <span className="text-[9px] text-muted-foreground uppercase font-medium">Generated Visual Evidence</span>
                                    </div>
                                </div>
                            )}
                            
                            {h.simulation_history && h.simulation_history.length > 0 && (
                                <div className="pt-2 border-t mt-2">
                                    <p className="text-[9px] font-bold uppercase text-muted-foreground mb-1">Rerun History ({h.simulation_history.length})</p>
                                    <div className="flex gap-1 overflow-x-auto pb-1">
                                        {h.simulation_history.map((prev, idx) => (
                                            <Badge key={idx} variant="outline" className="text-[8px] whitespace-nowrap opacity-60">
                                                Attempt {idx + 1}: {prev.success ? "Pass" : "Fail"}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        );
      })}
       {hypotheses.length === 0 && (
            <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                    <div key={i} className="h-24 w-full bg-muted/20 animate-pulse rounded-lg" />
                ))}
            </div>
        )}
    </div>
  );
}