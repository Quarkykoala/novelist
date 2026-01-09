import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export interface SoulMessage {
  soul: string;
  role?: string;
  text: string;
  timestamp: string;
  highlighted?: boolean;
}

interface SoulFeedProps {
  messages: SoulMessage[];
}

const ROLE_STYLES: Record<string, string> = {
  creative: "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-800",
  skeptic: "bg-slate-100 text-slate-800 border-slate-200 dark:bg-slate-800/50 dark:text-slate-200 dark:border-slate-700",
  methodical: "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-200 dark:border-blue-800",
  risk_taker: "bg-violet-100 text-violet-800 border-violet-200 dark:bg-violet-900/30 dark:text-violet-200 dark:border-violet-800",
  synthesizer: "bg-rose-100 text-red-800 border-red-200 dark:bg-rose-900/30 dark:text-rose-200 dark:border-rose-800",
};

const ROLE_ICONS: Record<string, string> = {
  creative: "🎨",
  skeptic: "🔍",
  methodical: "📐",
  risk_taker: "🚀",
  synthesizer: "⚗️",
};

export function SoulFeed({ messages }: SoulFeedProps) {
  return (
    <ScrollArea className="h-[calc(100vh-200px)] pr-4">
      <div className="flex flex-col gap-4 pb-4">
        {messages.map((msg, i) => {
          const role = msg.role?.toLowerCase() || "";
          const styles = ROLE_STYLES[role] || "bg-muted text-muted-foreground border-border";
          const icon = ROLE_ICONS[role] || "🤖";

          return (
            <div 
              key={i} 
              className={cn(
                "flex gap-3 p-3 rounded-lg border transition-all animate-in slide-in-from-bottom-2 fade-in",
                msg.highlighted ? "bg-accent/5 shadow-sm border-accent/20" : "bg-card border-border"
              )}
            >
              <div className={cn("w-8 h-8 rounded-full flex items-center justify-center text-sm border shrink-0", styles)}>
                {icon}
              </div>
              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs uppercase tracking-wider opacity-70">
                    {msg.soul}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
                  </span>
                </div>
                <p className="text-sm leading-relaxed text-foreground/90">
                  {msg.text}
                </p>
              </div>
            </div>
          );
        })}
        {messages.length === 0 && (
            <div className="text-center text-muted-foreground text-sm py-10">
                No soul activity yet. Start a session to see the debate.
            </div>
        )}
      </div>
    </ScrollArea>
  );
}
