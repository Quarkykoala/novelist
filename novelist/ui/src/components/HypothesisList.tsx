import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, XCircle, Link as LinkIcon, FileText, ExternalLink, Image as ImageIcon } from "lucide-react";

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
  };
}

interface HypothesisListProps {
  hypotheses: Hypothesis[];
  sourceMetadata?: Record<string, any>;
}

function getPlotFilename(path: string): string {
  try {
    return path.split(/[/\\]/).pop() || "";
  } catch (e) {
    return "";
  }
}

export function HypothesisList({ hypotheses, sourceMetadata = {} }: HypothesisListProps) {
  return (
    <div className="space-y-4">
      {hypotheses.map((h, i) => {
        const hasMechanism = h.rationale?.includes("→");
        const simResult = h.simulation_result;
        const isVerified = simResult?.success && simResult?.supports_hypothesis;

        return (
          <Card key={i} className="overflow-hidden animate-in fade-in slide-in-from-bottom-4" style={{ animationDelay: `${i * 100}ms` }}>
            <CardContent className="p-4 flex gap-4">
              <div className="text-4xl font-black text-muted/20 font-mono select-none">
                {i + 1}
              </div>
              <div className="flex-1 space-y-2">
                <p className="font-medium text-lg leading-snug">
                  {h.statement}
                </p>

                {/* Tags and Metadata */}
                <div className="flex flex-wrap gap-2 items-center">
                    {hasMechanism && (
                    <div className="inline-flex items-center gap-2 px-2 py-1 bg-violet-500/10 text-violet-600 dark:text-violet-300 rounded text-xs font-medium">
                        <LinkIcon className="w-3 h-3" />
                        {h.rationale?.split('.')[0]}
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
                    <div className="mt-4 rounded-md border bg-white p-4 flex flex-col gap-2">
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
                  <div className="mt-4 rounded-md border bg-muted/50 overflow-hidden">
                    <div className="p-2 border-b bg-muted/30 flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs font-semibold uppercase text-muted-foreground tracking-wider">
                            <FileText className="w-3 h-3" />
                            Lab Report
                        </div>
                        <Badge variant={isVerified ? "default" : "destructive"} className="gap-1">
                            {isVerified ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                            {isVerified ? "Verified In-Silico" : "Simulation Failed"}
                        </Badge>
                    </div>
                    
                    <div className="p-3">
                        <pre className="text-[10px] font-mono bg-background p-2 rounded border overflow-x-auto max-h-32">
                            {simResult.code}
                        </pre>
                        
                        {simResult.plot_path && (
                            <div className="mt-2 rounded overflow-hidden border">
                                <img 
                                    src={`/plots/${getPlotFilename(simResult.plot_path)}`} 
                                    alt="Simulation Plot" 
                                    className="w-full object-cover"
                                />
                            </div>
                        )}
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
