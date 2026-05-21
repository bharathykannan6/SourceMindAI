"use client";

import { GlassCard } from "@/components/ui/GlassCard";
import { AnimatedButton } from "@/components/ui/AnimatedButton";
import { 
  FileText, Zap, BrainCircuit, Activity, Upload, 
  Mic, Clock, ArrowUpRight, Shield, ArrowRight, Network
} from "lucide-react";
import { motion, Variants } from "framer-motion";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { fetchNotebooks, fetchDocuments, Notebook } from "@/lib/api";

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false);
  const { data: notebooks = [], isLoading: isLoadingNotebooks } = useQuery({
    queryKey: ['notebooks'],
    queryFn: () => fetchNotebooks(),
  });

  const { data: documents = [], isLoading: isLoadingDocuments } = useQuery({
    queryKey: ['documents'],
    queryFn: () => fetchDocuments(),
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  const containerVariants: Variants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const itemVariants: Variants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  if (!mounted) return null;

  return (
    <div className="p-8 max-w-[1600px] mx-auto w-full overflow-hidden">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end mb-8 gap-4">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
          <h2 className="font-display text-3xl font-bold mb-2 text-foreground">
            Welcome back, Dr. Aris
          </h2>
          <p className="text-muted-foreground flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
            </span>
            OpenNotebook synthesized 12 new insights while you were away.
          </p>
        </motion.div>
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex gap-3">
          <AnimatedButton variant="outline">
            <Upload className="w-4 h-4 mr-2" />
            Upload Data
          </AnimatedButton>
          <AnimatedButton variant="primary">
            <Mic className="w-4 h-4 mr-2" />
            Generate Podcast
          </AnimatedButton>
        </motion.div>
      </div>

      <motion.div 
        variants={containerVariants} 
        initial="hidden" 
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8"
      >
        {/* Analytics Cards */}
        <motion.div variants={itemVariants}>
          <GlassCard glowColor="primary" className="h-full">
            <div className="flex justify-between items-start mb-4">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
                <FileText className="w-5 h-5 text-primary" />
              </div>
              <span className="flex items-center text-xs font-medium text-green-400 bg-green-400/10 px-2 py-1 rounded-full">
                <ArrowUpRight className="w-3 h-3 mr-1" /> +12%
              </span>
            </div>
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-1">Total Documents</p>
            <h3 className="font-display text-3xl font-bold">
              {isLoadingDocuments ? <span className="animate-pulse">...</span> : documents.length}
            </h3>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard glowColor="accent" className="h-full">
            <div className="flex justify-between items-start mb-4">
              <div className="w-10 h-10 rounded-xl bg-accent/10 flex items-center justify-center">
                <Clock className="w-5 h-5 text-accent" />
              </div>
              <span className="flex items-center text-xs font-medium text-green-400 bg-green-400/10 px-2 py-1 rounded-full">
                <ArrowUpRight className="w-3 h-3 mr-1" /> +5h
              </span>
            </div>
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-1">Hours Saved</p>
            <h3 className="font-display text-3xl font-bold">42.5</h3>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard glowColor="secondary" className="h-full">
            <div className="flex justify-between items-start mb-4">
              <div className="w-10 h-10 rounded-xl bg-secondary/10 flex items-center justify-center">
                <BrainCircuit className="w-5 h-5 text-secondary" />
              </div>
              <span className="flex items-center text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded-full">
                94% Accuracy
              </span>
            </div>
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-1">Insights Generated</p>
            <h3 className="font-display text-3xl font-bold">342</h3>
          </GlassCard>
        </motion.div>

        <motion.div variants={itemVariants}>
          <GlassCard className="h-full relative overflow-hidden" glowColor="primary">
            <div className="absolute -right-4 -top-4 opacity-10">
              <Activity className="w-32 h-32 text-primary" />
            </div>
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-wider mb-2">Token Usage</p>
            <h3 className="font-display text-3xl font-bold mb-4">8.2M <span className="text-sm text-muted-foreground font-body font-normal">/ 10M</span></h3>
            <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: "82%" }}
                transition={{ duration: 1, delay: 0.5 }}
                className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
              />
            </div>
          </GlassCard>
        </motion.div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Main Content Area */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="lg:col-span-2 space-y-8"
        >
          {/* Recent Notebooks */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display text-xl font-bold">Recent Notebooks</h3>
              <AnimatedButton variant="ghost" size="sm">View All <ArrowRight className="w-4 h-4 ml-1" /></AnimatedButton>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              {isLoadingNotebooks ? (
                <div className="col-span-2 flex justify-center items-center h-24">
                  <span className="text-muted-foreground animate-pulse flex items-center gap-2">
                    Loading notebooks...
                  </span>
                </div>
              ) : notebooks.length > 0 ? (
                notebooks.slice(0, 4).map((notebook: Notebook) => (
                  <GlassCard key={notebook.id} className="p-0 overflow-hidden group cursor-pointer" glowColor="primary">
                    <div className="h-24 bg-surface/50 border-b border-white/5 relative p-4 flex items-end">
                      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent"></div>
                      <h4 className="font-display text-lg font-bold relative z-10 group-hover:text-primary transition-colors">{notebook.title}</h4>
                    </div>
                    <div className="p-4">
                      <p className="text-sm text-muted-foreground mb-4 line-clamp-2">{notebook.description || "No description provided."}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground flex items-center gap-1"><FileText className="w-3 h-3" /> Workspace: {notebook.workspace_id.slice(0, 8)}</span>
                      </div>
                    </div>
                  </GlassCard>
                ))
              ) : (
                <div className="col-span-2 flex justify-center items-center h-24 glass rounded-xl">
                  <p className="text-muted-foreground text-sm">No notebooks found. Create one to get started.</p>
                </div>
              )}
            </div>
          </div>

          {/* Activity Chart Placeholder */}
          <GlassCard className="h-72 flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-display text-lg font-bold">Knowledge Synthesis Trend</h3>
              <div className="flex gap-4">
                <span className="flex items-center gap-2 text-xs text-muted-foreground"><div className="w-2 h-2 rounded-full bg-primary"></div> Papers Parsed</span>
                <span className="flex items-center gap-2 text-xs text-muted-foreground"><div className="w-2 h-2 rounded-full bg-accent"></div> Insights Found</span>
              </div>
            </div>
            
            <div className="flex-1 flex items-end justify-between gap-2 md:gap-4 relative px-2 pb-2">
              {/* Fake animated bars */}
              {[40, 65, 30, 85, 55, 45, 70, 90, 60, 40].map((h, i) => (
                <div key={i} className="flex-1 flex flex-col justify-end gap-1 group h-full relative">
                  <motion.div 
                    initial={{ height: 0 }}
                    animate={{ height: `${h * 0.4}%` }}
                    transition={{ duration: 1, delay: 0.1 * i }}
                    className="w-full bg-primary/40 rounded-t-sm hover:bg-primary/60 transition-colors"
                  />
                  <motion.div 
                    initial={{ height: 0 }}
                    animate={{ height: `${h}%` }}
                    transition={{ duration: 1, delay: 0.1 * i }}
                    className="w-full bg-accent/40 rounded-t-sm hover:bg-accent/60 transition-colors"
                  />
                </div>
              ))}
            </div>
          </GlassCard>

        </motion.div>

        {/* Right Sidebar Area */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="space-y-6"
        >
          {/* AI Activity Feed */}
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-display text-lg font-bold">Live AI Insights</h3>
            <Activity className="w-4 h-4 text-primary animate-pulse" />
          </div>

          <div className="space-y-4">
            <GlassCard className="p-4 border-l-4 border-l-primary relative overflow-hidden" hoverEffect={false}>
              <div className="absolute top-0 right-0 p-4 opacity-5"><BrainCircuit className="w-16 h-16" /></div>
              <div className="flex items-start gap-3 relative z-10">
                <div className="mt-1"><Zap className="w-4 h-4 text-primary" /></div>
                <div>
                  <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase tracking-wider">Correlation Found</div>
                  <p className="text-sm font-medium leading-relaxed">OpenNotebook detected a structural similarity between "Project Helios" and "Thermal Diffusion Alpha".</p>
                  <AnimatedButton variant="ghost" size="sm" className="mt-2 h-auto py-1 px-2 -ml-2 text-xs text-primary">
                    Explore Connection <ArrowRight className="w-3 h-3 ml-1" />
                  </AnimatedButton>
                </div>
              </div>
            </GlassCard>

            <GlassCard className="p-4 border-l-4 border-l-secondary relative overflow-hidden" hoverEffect={false}>
              <div className="flex items-start gap-3 relative z-10">
                <div className="mt-1"><FileText className="w-4 h-4 text-secondary" /></div>
                <div>
                  <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase tracking-wider">Document Processing</div>
                  <p className="text-sm font-medium leading-relaxed">3 new papers matching your "Vector Databases" workspace have been vectorized and added to the graph.</p>
                </div>
              </div>
            </GlassCard>

            <GlassCard className="p-4 border-l-4 border-l-accent relative overflow-hidden" hoverEffect={false}>
              <div className="flex items-start gap-3 relative z-10">
                <div className="mt-1"><Shield className="w-4 h-4 text-accent" /></div>
                <div>
                  <div className="text-[10px] font-mono text-muted-foreground mb-1 uppercase tracking-wider">Security Scan</div>
                  <p className="text-sm font-medium leading-relaxed">PII redaction completed on recent lab uploads. 14 items scrubbed.</p>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Mini Knowledge Graph Widget */}
          <GlassCard className="h-64 flex flex-col items-center justify-center text-center relative overflow-hidden mt-6" glowColor="primary">
            <div className="absolute inset-0 opacity-20 pointer-events-none bg-[radial-gradient(ellipse_at_center,rgba(192,193,255,0.4)_0%,transparent_70%)]"></div>
            <motion.div 
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="mb-4"
            >
              <Network className="w-12 h-12 text-primary opacity-80" />
            </motion.div>
            <h4 className="font-display font-bold relative z-10">Knowledge Orbit</h4>
            <p className="text-xs text-muted-foreground mt-2 mb-4 relative z-10 max-w-[200px]">Interact with your latest research nodes in 3D space.</p>
            <AnimatedButton variant="outline" size="sm" className="relative z-10">Launch Visualizer</AnimatedButton>
          </GlassCard>

        </motion.div>
      </div>
    </div>
  );
}
