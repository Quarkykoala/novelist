import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FileText, ExternalLink } from "lucide-react";
import type { Hypothesis } from "@/components/HypothesisList";

interface EvidenceBoardProps {
  hypotheses: Hypothesis[];
  sourceMetadata: Record<string, any>;
}

export function EvidenceBoard({ hypotheses, sourceMetadata }: EvidenceBoardProps) {
  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 border-b">
        <CardTitle className="text-sm uppercase tracking-wider flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary" />
          Evidence Board
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto p-4 space-y-6">
        {hypotheses.length === 0 && (
          <div className="text-center text-muted-foreground text-xs py-10">
            No hypotheses generated yet.
          </div>
        )}
        {hypotheses.map((h) => (
          <div key={h.id} className="space-y-2">
            <div className="flex items-start gap-2">
              <Badge variant="outline" className="shrink-0">{h.id.slice(0,6)}</Badge>
              <p className="text-xs font-medium leading-tight">{h.statement}</p>
            </div>
            
            <div className="pl-14 space-y-1">
              {h.supported_facts && h.supported_facts.length > 0 && (
                <div className="text-[10px] space-y-1">
                  <p className="font-semibold text-emerald-700 uppercase">Supported Facts</p>
                  {h.supported_facts.slice(0, 3).map((fact, idx) => (
                    <div key={`${h.id}-sf-${idx}`} className="bg-emerald-500/5 p-1.5 rounded">
                      {fact}
                    </div>
                  ))}
                </div>
              )}

              {h.novel_inference && (
                <div className="text-[10px] space-y-1">
                  <p className="font-semibold text-indigo-700 uppercase">Novel Inference</p>
                  <div className="bg-indigo-500/5 p-1.5 rounded">{h.novel_inference}</div>
                </div>
              )}

              {h.unsupported_parts && h.unsupported_parts.length > 0 && (
                <div className="text-[10px] space-y-1">
                  <p className="font-semibold text-amber-700 uppercase">Unsupported Parts</p>
                  {h.unsupported_parts.slice(0, 3).map((part, idx) => (
                    <div key={`${h.id}-up-${idx}`} className="bg-amber-500/10 p-1.5 rounded">
                      {part}
                    </div>
                  ))}
                </div>
              )}

              {h.supporting_papers && h.supporting_papers.length > 0 ? (
                h.supporting_papers.slice(0, 5).map((paperId) => {
                  const paper = sourceMetadata[paperId];
                  return (
                    <div key={paperId} className="flex items-center gap-2 text-[10px] bg-muted/50 p-1.5 rounded border border-transparent hover:border-border transition-colors">
                      <FileText className="w-3 h-3 text-muted-foreground" />
                      <div className="flex-1 min-w-0">
                        <span className="font-semibold text-foreground">{paperId}</span>
                        {paper && (
                          <span className="text-muted-foreground ml-1 truncate block">
                            {paper.title}
                          </span>
                        )}
                      </div>
                      {paper?.abs_url && (
                        <a 
                          href={paper.abs_url} 
                          target="_blank" 
                          rel="noreferrer"
                          className="text-primary hover:text-primary/80"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </div>
                  );
                })
              ) : (
                <span className="text-[10px] text-red-500 italic">Ungrounded - no pulled-corpus citations linked.</span>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
