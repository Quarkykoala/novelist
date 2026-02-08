import { Button } from "@/components/ui/button";
import { Bell, Moon, Sun } from "lucide-react";
import { Switch } from "@/components/ui/switch"; // I need to install switch!
import { Label } from "@/components/ui/label"; // And label!
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface HeaderProps {
  onThemeToggle: () => void;
  isDarkMode: boolean;
}

export function Header({ onThemeToggle, isDarkMode }: HeaderProps) {
  return (
    <header className="h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center justify-between px-6">
      <div>
        <h1 className="font-semibold text-lg">Scientific Hypothesis Synthesizer</h1>
        <p className="text-xs text-muted-foreground">Automated Generation and Evaluation</p>
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-secondary/50 rounded-full border">
            <Label htmlFor="superprompt" className="text-xs font-medium cursor-pointer">SuperPrompt Mode</Label>
            <Switch id="superprompt" defaultChecked />
        </div>

        <Button variant="ghost" size="icon">
          <Bell className="w-4 h-4" />
        </Button>
        
        <Button variant="ghost" size="icon" onClick={onThemeToggle}>
          {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </Button>

        <Avatar className="w-8 h-8">
            <AvatarImage src="" />
            <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
}
