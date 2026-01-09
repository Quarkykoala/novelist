import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { 
  LayoutDashboard, 
  FlaskConical, 
  FolderOpen, 
  History, 
  Settings, 
  Hexagon 
} from "lucide-react";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";

interface SidebarProps {
  className?: string;
  activePage: string;
  onNavigate: (page: string) => void;
}

export function Sidebar({ className, activePage, onNavigate }: SidebarProps) {
  const [apiStatus, setApiStatus] = useState<"connected" | "offline" | "error">("offline");

  useEffect(() => {
    const check = async () => {
      const isConnected = await api.checkStatus();
      setApiStatus(isConnected ? "connected" : "offline");
    };
    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "lab", label: "Hypothesis Lab", icon: FlaskConical },
    { id: "knowledge", label: "Knowledge", icon: FolderOpen },
    { id: "history", label: "History", icon: History },
    { id: "settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside className={cn("w-64 border-r bg-background flex flex-col h-screen", className)}>
      <div className="p-6 flex items-center gap-2 border-b">
        <Hexagon className="w-6 h-6 text-primary fill-primary/20" />
        <span className="font-bold text-lg tracking-tight">Novelist</span>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => (
          <Button
            key={item.id}
            variant={activePage === item.id ? "secondary" : "ghost"}
            className={cn(
              "w-full justify-start gap-3",
              activePage === item.id && "bg-secondary"
            )}
            onClick={() => onNavigate(item.id)}
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </Button>
        ))}
      </nav>

      <div className="p-4 border-t">
        <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <span className={cn(
            "w-2 h-2 rounded-full",
            apiStatus === "connected" ? "bg-green-500" : "bg-red-500"
          )} />
          {apiStatus === "connected" ? "API Connected" : "API Offline"}
        </div>
      </div>
    </aside>
  );
}
