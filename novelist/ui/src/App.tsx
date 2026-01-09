import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { Dashboard } from "@/pages/Dashboard";

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check system preference or localStorage
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        setIsDarkMode(true);
        document.documentElement.classList.add("dark");
    }
  }, []);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    if (!isDarkMode) {
        document.documentElement.classList.add("dark");
    } else {
        document.documentElement.classList.remove("dark");
    }
  };

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Sidebar 
        activePage={activePage} 
        onNavigate={setActivePage} 
      />
      
      <div className="flex-1 flex flex-col min-w-0">
        <Header 
            onThemeToggle={toggleTheme} 
            isDarkMode={isDarkMode} 
        />
        
        <main className="flex-1 bg-muted/20 relative">
            {activePage === "dashboard" && <Dashboard />}
            {activePage !== "dashboard" && (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                    Work in progress...
                </div>
            )}
        </main>
      </div>
    </div>
  );
}

export default App;