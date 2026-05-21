"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useQueryClient, useMutation } from "@tanstack/react-query";
import { createNotebook } from "@/lib/api";
import {
  LayoutDashboard,
  FileEdit,
  Files,
  MessageSquare,
  Mic,
  Network,
  Settings,
  Shield,
  LogOut,
  Plus
} from "lucide-react";
import { AnimatedButton } from "../ui/AnimatedButton";

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();

  const createNotebookMutation = useMutation({
    mutationFn: (name: string) => createNotebook(name),
    onSuccess: (newNotebook) => {
      // Invalidate queries for notebooks so the list refreshes
      queryClient.invalidateQueries({ queryKey: ['notebooks'] });
      // Redirect to the workspace page and select the new notebook
      router.push(`/workspace?notebookId=${newNotebook.id}`);
    },
    onError: (error: any) => {
      console.error("Failed to create notebook:", error);
      alert("Failed to create notebook: " + (error.message || "Unknown error"));
    }
  });

  const handleCreateNotebook = () => {
    const name = window.prompt("Enter new notebook name:");
    if (name && name.trim()) {
      createNotebookMutation.mutate(name.trim());
    }
  };

  const links = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { name: "Workspace", href: "/workspace", icon: FileEdit },
    { name: "Documents", href: "/documents", icon: Files },
    { name: "AI Chat", href: "/chat", icon: MessageSquare },
    { name: "Podcast Studio", href: "/podcast", icon: Mic },
    { name: "Knowledge Graph", href: "/graph", icon: Network },
  ];

  return (
    <aside className="w-[280px] h-screen fixed left-0 top-0 border-r border-white/10 bg-surface/40 backdrop-blur-[20px] flex flex-col p-6 z-50">
      <div className="mb-8 px-2">
        <Link href="/">
          <h1 className="font-display text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent mb-1">
            OpenNotebook
          </h1>
        </Link>
        <p className="font-mono text-xs text-muted-foreground uppercase tracking-widest opacity-70">
          Research Engine
        </p>
      </div>

      <AnimatedButton 
        variant="primary" 
        className="w-full mb-8" 
        size="md"
        onClick={handleCreateNotebook}
        disabled={createNotebookMutation.isPending}
      >
        <Plus className="w-5 h-5" />
        {createNotebookMutation.isPending ? "Creating..." : "New Notebook"}
      </AnimatedButton>

      <nav className="flex-1 space-y-1.5 overflow-y-auto no-scrollbar">
        {links.map((link) => {
          const isActive = pathname === link.href;
          const Icon = link.icon;
          return (
            <Link
              key={link.name}
              href={link.href}
              className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group",
                isActive
                  ? "bg-white/10 border-l-2 border-primary text-primary font-medium shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/5"
              )}
            >
              <Icon
                className={cn(
                  "w-5 h-5 transition-transform duration-300",
                  isActive ? "scale-110 drop-shadow-[0_0_8px_rgba(192,193,255,0.5)]" : "group-hover:scale-110"
                )}
              />
              <span className="font-body text-sm">{link.name}</span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-6 border-t border-white/10 space-y-2">
        <Link
          href="/settings"
          className="flex items-center gap-3 px-4 py-2.5 text-muted-foreground hover:text-foreground transition-colors rounded-xl hover:bg-white/5"
        >
          <Settings className="w-5 h-5" />
          <span className="font-body text-sm">Settings</span>
        </Link>
        <Link
          href="/admin"
          className="flex items-center gap-3 px-4 py-2.5 text-muted-foreground hover:text-foreground transition-colors rounded-xl hover:bg-white/5"
        >
          <Shield className="w-5 h-5" />
          <span className="font-body text-sm">Admin</span>
        </Link>
        <Link
          href="/login"
          className="flex items-center gap-3 px-4 py-2.5 text-muted-foreground hover:text-red-400 transition-colors rounded-xl hover:bg-red-500/10"
        >
          <LogOut className="w-5 h-5" />
          <span className="font-body text-sm">Logout</span>
        </Link>
      </div>
    </aside>
  );
}
