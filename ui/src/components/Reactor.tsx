import { cn } from "@/lib/utils";

type PhaseRecord = {
  phase: string;
  detail?: string;
  timestamp?: string;
};

interface ReactorProps {
  isActive: boolean;
  status: string;
  phase: string;
  history?: PhaseRecord[];
  statusDetail?: string;
}

const STAGES = [
  { key: "queued", label: "Queued" },
  { key: "forging", label: "Forging" },
  { key: "mapping", label: "Mapping" },
  { key: "debating", label: "Debating" },
  { key: "verifying", label: "Verifying" },
  { key: "complete", label: "Complete" },
  { key: "error", label: "Error" },
];

export function Reactor({ isActive, status, phase, history = [], statusDetail }: ReactorProps) {
  const showErrorStage = phase === "error" || history.some((entry) => entry.phase === "error");
  const visibleStages = showErrorStage
    ? STAGES
    : STAGES.filter((stage) => stage.key !== "error");
  const currentIndex = visibleStages.findIndex((stage) => stage.key === phase);
  return (
    <div className="relative flex flex-col items-center justify-center py-8">
      <div className="relative w-48 h-48 flex items-center justify-center">
        {/* Glow */}
        <div className={cn(
          "absolute inset-0 bg-primary/20 blur-3xl rounded-full transition-all duration-1000",
          isActive ? "opacity-100 scale-125" : "opacity-0 scale-75"
        )} />

        {/* Outer Ring */}
        <div className={cn(
          "absolute w-full h-full border-2 border-primary/30 rounded-[40%] transition-all duration-1000",
          isActive ? "animate-spin-slow border-primary/50" : "scale-90 opacity-50"
        )} />

        {/* Inner Ring */}
        <div className={cn(
          "absolute w-3/4 h-3/4 border-2 border-primary/40 rounded-[60%] transition-all duration-1000",
          isActive ? "animate-spin-reverse border-primary/60" : "scale-90 opacity-50"
        )} />

        {/* Core */}
        <div className={cn(
          "relative w-16 h-16 bg-gradient-to-br from-primary to-violet-600 transition-all duration-1000",
          isActive ? "animate-morph animate-pulse-glow" : "rounded-full opacity-50 grayscale scale-75"
        )}>
            <div className="absolute inset-2 bg-white/20 blur-sm rounded-full" />
        </div>
      </div>
      
      <div className={cn(
        "mt-6 text-sm font-medium transition-colors font-mono uppercase tracking-wider",
        isActive ? "text-primary animate-pulse" : "text-muted-foreground"
      )}>
        {status}
      </div>
      {statusDetail && (
        <div className="text-xs text-muted-foreground mt-1 text-center max-w-sm">
          {statusDetail}
        </div>
      )}
      <div className="w-full mt-8 px-6">
        <div className="flex items-center justify-between">
          {visibleStages.map((stage, idx) => {
            const stageReached = currentIndex >= idx && currentIndex !== -1;
            const detail = [...history].reverse().find((entry: PhaseRecord) => entry.phase === stage.key)?.detail;
            return (
              <div key={stage.key} className="flex flex-col items-center text-center flex-1">
                <div
                  className={cn(
                    "w-8 h-8 rounded-full border flex items-center justify-center text-xs font-semibold transition-colors",
                    stageReached
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-muted-foreground"
                  )}
                >
                  {idx + 1}
                </div>
                <span className="mt-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                  {stage.label}
                </span>
                {detail && (
                  <span className="mt-1 text-[10px] text-muted-foreground/70 line-clamp-2">
                    {detail}
                  </span>
                )}
              </div>
          );
          })}
        </div>
      </div>
    </div>
  );
}
