import { cn } from "@/lib/utils";

interface ReactorProps {
  isActive: boolean;
  status: string;
}

export function Reactor({ isActive, status }: ReactorProps) {
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
    </div>
  );
}
