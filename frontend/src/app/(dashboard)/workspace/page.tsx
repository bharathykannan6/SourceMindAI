"use client";

import { useState, useRef, useEffect, Suspense } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { GlassCard } from "@/components/ui/GlassCard";
import { AnimatedButton } from "@/components/ui/AnimatedButton";
import { 
  FileText, Search, Folder, Plus, Bot, ArrowUpRight, 
  Paperclip, Mic, Send, Network, Loader2, Sparkles, X, LayoutTemplate, Trash2,
  CheckSquare, CheckCheck, Filter, RotateCcw, Pencil
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { 
  fetchDocuments, 
  uploadDocument, 
  fetchNotebooks, 
  createNotebook,
  renameNotebook,
  deleteNotebook,
  sendChatMessage, 
  ingestText,
  deleteDocument,
  Document, 
  Notebook, 
  Citation 
} from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

function WorkspaceContent() {
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const notebookIdParam = searchParams.get("notebookId");

  const [isTyping, setIsTyping] = useState(false);
  const [showGraph, setShowGraph] = useState(true);
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);
  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set());
  const [chatInput, setChatInput] = useState("");
  const [isPasteModalOpen, setIsPasteModalOpen] = useState(false);
  const [pasteTitle, setPasteTitle] = useState("");
  const [pasteText, setPasteText] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "I've fully ingested your workspace documents. The vectors are mapped and the knowledge graph is ready. How would you like to explore your notes and sources today?",
    }
  ]);

  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch notebooks (folders)
  const { data: notebooks = [], isLoading: isLoadingNotebooks } = useQuery({
    queryKey: ['notebooks'],
    queryFn: () => fetchNotebooks(),
  });

  // Sync state selection when URL parameter changes
  useEffect(() => {
    if (notebookIdParam) {
      setSelectedNotebookId(notebookIdParam);
    }
  }, [notebookIdParam]);

  // Set default selected notebook only if there's no URL param and no active selection
  useEffect(() => {
    if (notebooks.length > 0 && !selectedNotebookId && !notebookIdParam) {
      setSelectedNotebookId(notebooks[0].id);
    }
  }, [notebooks, selectedNotebookId, notebookIdParam]);

  // Helper to select notebook and update the URL
  const handleSelectNotebook = (id: string) => {
    setSelectedNotebookId(id);
    const params = new URLSearchParams(searchParams.toString());
    params.set("notebookId", id);
    router.replace(`${pathname}?${params.toString()}`);
  };

  // Fetch documents for the selected notebook
  // Poll every 3 seconds while any document is pending/processing
  const { data: documents = [], isLoading: isLoadingDocuments } = useQuery({
    queryKey: ['documents', selectedNotebookId],
    queryFn: () => fetchDocuments(selectedNotebookId || undefined),
    enabled: !!selectedNotebookId,
    refetchInterval: (query) => {
      const docs = query.state.data as Document[] | undefined;
      if (!docs) return 3000;
      const hasActive = docs.some(
        (d) => d.status === 'pending' || d.status === 'processing'
      );
      return hasActive ? 3000 : false;
    },
  });

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Auto-resize textarea
  useEffect(() => {
    const handleInput = () => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
      }
    };
    
    const ta = textareaRef.current;
    if (ta) ta.addEventListener('input', handleInput);
    return () => { if (ta) ta.removeEventListener('input', handleInput); };
  }, []);

  // Upload Mutation
  const uploadMutation = useMutation({
    mutationFn: (file: File) =>
      uploadDocument(file, selectedNotebookId || undefined, (pct) =>
        setUploadProgress(pct)
      ),
    onSuccess: () => {
      setUploadError(null);
      setUploadProgress(null);
      queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
    },
    onError: (error: any) => {
      setUploadProgress(null);
      const msg = error?.response?.data?.detail || error?.message || "Upload failed. Check that the backend is running and MinIO is accessible.";
      setUploadError(msg);
      console.error("Upload failed", error);
      queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
    }
  });

  // Paste Text Mutation
  const pasteMutation = useMutation({
    mutationFn: ({ text, title }: { text: string; title: string }) => 
      ingestText(text, title, selectedNotebookId || undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
      setIsPasteModalOpen(false);
      setPasteText("");
      setPasteTitle("");
    },
    onError: (error) => {
      console.error("Pasted text ingestion failed", error);
      alert("Failed to ingest pasted text: " + (error as any).message);
    }
  });

  // Delete Document Mutation
  const deleteDocumentMutation = useMutation({
    mutationFn: (documentId: string) => deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
    },
    onError: (error) => {
      console.error("Failed to delete document", error);
      alert("Failed to delete document.");
    }
  });

  const [renamingNotebookId, setRenamingNotebookId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  // Delete Notebook Mutation
  const deleteNotebookMutation = useMutation({
    mutationFn: (notebookId: string) => deleteNotebook(notebookId),
    onSuccess: (_, notebookId) => {
      queryClient.invalidateQueries({ queryKey: ['notebooks'] });
      // If deleted notebook was selected, clear selection
      if (selectedNotebookId === notebookId) {
        setSelectedNotebookId(null);
        const params = new URLSearchParams(searchParams.toString());
        params.delete('notebookId');
        router.replace(`${pathname}?${params.toString()}`);
      }
    },
    onError: (error: any) => {
      console.error('Failed to delete notebook', error);
      alert('Failed to delete notebook: ' + (error?.response?.data?.detail || error.message));
    }
  });

  // Rename Notebook Mutation
  const renameNotebookMutation = useMutation({
    mutationFn: ({ id, name }: { id: string; name: string }) => renameNotebook(id, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notebooks'] });
      setRenamingNotebookId(null);
      setRenameValue("");
    },
    onError: (error: any) => {
      console.error('Failed to rename notebook', error);
      alert('Failed to rename: ' + (error?.response?.data?.detail || error.message));
    }
  });

  // Create Notebook (Folder) Mutation
  const createNotebookMutation = useMutation({
    mutationFn: (name: string) => createNotebook(name),
    onSuccess: (newNotebook) => {
      queryClient.invalidateQueries({ queryKey: ['notebooks'] });
      setSelectedNotebookId(newNotebook.id);
      
      // Update URL search parameters
      const params = new URLSearchParams(searchParams.toString());
      params.set("notebookId", newNotebook.id);
      router.replace(`${pathname}?${params.toString()}`);
    },
    onError: (error: any) => {
      console.error("Failed to create folder", error);
      alert("Failed to create folder: " + (error.message || "Unknown error"));
    }
  });

  const handleCreateFolder = () => {
    const name = window.prompt("Enter new folder/notebook name:");
    if (name && name.trim()) {
      createNotebookMutation.mutate(name.trim());
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setUploadError(null);
      // Upload files sequentially — not in parallel — to avoid connection conflicts
      for (const file of Array.from(files)) {
        await uploadMutation.mutateAsync(file).catch(() => {});
      }
    }
    e.target.value = "";
  };

  const toggleDocSelection = (docId: string) => {
    setSelectedDocIds(prev => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  };

  // Reset doc selection when notebook changes
  useEffect(() => {
    setSelectedDocIds(new Set());
  }, [selectedNotebookId]);

  const handleSendMessage = async () => {
    if (!chatInput.trim() || !selectedNotebookId) return;

    const userMsg = chatInput.trim();
    setChatInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    // Append user message
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsTyping(true);

    try {
      const response = await sendChatMessage(
        userMsg,
        selectedNotebookId,
        selectedDocIds.size > 0 ? Array.from(selectedDocIds) : undefined
      );
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: response.response,
        citations: response.citations
      }]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: "Sorry, I encountered an error while processing your request. Please check if your backend server and external AI models are correctly configured."
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        role: "assistant",
        content: "I've fully ingested your workspace documents. The vectors are mapped and the knowledge graph is ready. How would you like to explore your notes and sources today?",
      }
    ]);
  };

  const activeNotebook = notebooks.find(n => n.id === selectedNotebookId);

  return (
    <div className="flex h-[calc(100vh-64px)] overflow-hidden w-full bg-background relative selection:bg-primary/30 selection:text-primary">
      
      {/* Left Pane: Document Explorer */}
      <div className="w-[300px] flex-shrink-0 border-r border-white/10 bg-surface/30 backdrop-blur-md flex flex-col hidden lg:flex">
        <div className="p-4 border-b border-white/10 flex items-center justify-between">
          <h3 className="font-display font-semibold text-foreground">Active Sources</h3>
          <AnimatedButton 
            variant="ghost" 
            size="icon" 
            className="h-8 w-8 text-primary"
            onClick={handleCreateFolder}
            disabled={createNotebookMutation.isPending}
          >
            {createNotebookMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Plus className="w-4 h-4" />
            )}
          </AnimatedButton>
        </div>

        <div className="p-4 flex-1 overflow-y-auto no-scrollbar space-y-6">
          {/* Folders (Notebooks) */}
          <div className="space-y-1">
            <div className="text-xs font-mono text-muted-foreground mb-2 px-2 uppercase tracking-wider">Notebooks</div>
            {isLoadingNotebooks ? (
              <div className="text-xs text-muted-foreground px-2 py-2 flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" /> Loading folders...
              </div>
            ) : notebooks.length === 0 ? (
              <div className="text-xs text-muted-foreground px-2 py-2">No folders yet. Click '+' to add.</div>
            ) : (
              notebooks.map((nb) => (
                <div
                  key={nb.id}
                  onClick={() => renamingNotebookId !== nb.id && handleSelectNotebook(nb.id)}
                  className={cn(
                    "flex items-center gap-2 px-2 py-1.5 text-sm cursor-pointer transition-all rounded-lg border group",
                    selectedNotebookId === nb.id 
                      ? "text-primary bg-primary/10 border-primary/20 font-medium" 
                      : "text-muted-foreground hover:text-foreground hover:bg-white/5 border-transparent"
                  )}
                >
                  <Folder className="w-4 h-4 flex-shrink-0" />
                  {renamingNotebookId === nb.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && renameValue.trim()) {
                          renameNotebookMutation.mutate({ id: nb.id, name: renameValue.trim() });
                        }
                        if (e.key === 'Escape') {
                          setRenamingNotebookId(null);
                          setRenameValue("");
                        }
                      }}
                      onBlur={() => {
                        if (renameValue.trim() && renameValue.trim() !== nb.title) {
                          renameNotebookMutation.mutate({ id: nb.id, name: renameValue.trim() });
                        } else {
                          setRenamingNotebookId(null);
                        }
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="flex-1 bg-transparent border-b border-primary/50 outline-none text-foreground text-sm py-0.5 min-w-0"
                    />
                  ) : (
                    <span className="truncate flex-1">{nb.title}</span>
                  )}
                  {renamingNotebookId !== nb.id && (
                    <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 flex-shrink-0">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setRenamingNotebookId(nb.id);
                          setRenameValue(nb.title);
                        }}
                        className="p-1 rounded hover:bg-white/10 text-muted-foreground hover:text-primary transition-colors"
                        title="Rename notebook"
                      >
                        <Pencil className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteNotebookMutation.mutate(nb.id);
                        }}
                        disabled={deleteNotebookMutation.isPending}
                        className="p-1 rounded hover:bg-red-500/10 text-muted-foreground hover:text-red-400 transition-colors"
                        title="Delete notebook"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Files (Documents) */}
          <div>
            <div className="text-xs font-mono text-muted-foreground mb-3 px-2 uppercase tracking-wider flex items-center justify-between">
              <span>Vectorized Documents</span>
              {documents.length > 0 && (
                <button
                  onClick={() => {
                    const doneDocs = documents.filter(d => d.status === 'done');
                    if (selectedDocIds.size === doneDocs.length && doneDocs.length > 0) {
                      setSelectedDocIds(new Set());
                    } else {
                      setSelectedDocIds(new Set(doneDocs.map(d => d.id)));
                    }
                  }}
                  className="text-[10px] text-primary hover:text-primary/80 transition-colors flex items-center gap-1 font-sans normal-case tracking-normal"
                  title={selectedDocIds.size > 0 ? "Clear selection" : "Select all ready docs"}
                >
                  {selectedDocIds.size > 0 ? (
                    <><CheckCheck className="w-3 h-3" /> Clear</>
                  ) : (
                    <><CheckSquare className="w-3 h-3" /> All</>
                  )}
                </button>
              )}
            </div>
            <div className="space-y-2">
              {!selectedNotebookId ? (
                <div className="text-center text-xs text-muted-foreground py-4">Select a folder to view documents.</div>
              ) : isLoadingDocuments ? (
                <div className="text-center text-xs text-muted-foreground py-4 flex items-center justify-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" /> Loading...
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center text-xs text-muted-foreground py-4">No documents in this notebook yet.</div>
              ) : (
                documents.map((doc: Document) => (
                  <div
                    key={doc.id}
                    className={cn(
                      "flex items-start gap-2 p-3 rounded-xl border transition-all group cursor-pointer",
                      selectedDocIds.has(doc.id)
                        ? "border-primary/40 bg-primary/10"
                        : "border-transparent hover:bg-white/5"
                    )}
                    onClick={() => doc.status === 'done' && toggleDocSelection(doc.id)}
                  >
                    {/* Checkbox */}
                    <div className={cn(
                      "mt-0.5 w-4 h-4 rounded border flex-shrink-0 flex items-center justify-center transition-all",
                      doc.status === 'done'
                        ? selectedDocIds.has(doc.id)
                          ? "border-primary bg-primary"
                          : "border-white/20 group-hover:border-primary/50"
                        : "border-white/10 opacity-30"
                    )}>
                      {selectedDocIds.has(doc.id) && (
                        <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                    <FileText className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                    <div className="overflow-hidden flex-1 min-w-0">
                      <p className="text-sm font-medium text-muted-foreground truncate group-hover:text-foreground transition-colors">{doc.file_name}</p>
                      <div className="flex gap-2 mt-1">
                        <span className={cn(
                          "text-[10px] px-1.5 py-0.5 rounded bg-surface border uppercase font-mono",
                          doc.status === "done" ? "border-green-500/20 text-green-400" :
                          doc.status === "processing" ? "border-yellow-500/20 text-yellow-400" :
                          doc.status === "error" ? "border-red-500/20 text-red-400" :
                          "border-white/10 text-muted-foreground"
                        )}>
                          {doc.status}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteDocumentMutation.mutate(doc.id);
                      }}
                      disabled={deleteDocumentMutation.isPending}
                      className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg hover:bg-red-500/10 text-muted-foreground hover:text-red-400 flex-shrink-0"
                      title="Delete document"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Upload Zone */}
        <div className="p-4 border-t border-white/10">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="hidden"
            multiple
            accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.md,.mp3,.wav,.m4a,.ogg,.flac,.webm,.aac,.png,.jpg,.jpeg,.tiff,.bmp"
          />
          <div 
            onClick={() => {
              if (selectedNotebookId) {
                fileInputRef.current?.click();
              } else {
                alert("Please select or create a notebook folder first.");
              }
            }}
            className={cn(
              "border border-dashed border-white/20 rounded-xl p-4 text-center cursor-pointer transition-all",
              !selectedNotebookId && "opacity-50 cursor-not-allowed",
              uploadMutation.isPending ? "border-primary/50 bg-primary/10" : "hover:border-primary/50 hover:bg-primary/5"
            )}
          >
            <div className="w-8 h-8 rounded-full bg-surface border border-white/10 flex items-center justify-center mx-auto mb-2">
              {uploadMutation.isPending ? (
                <Loader2 className="w-4 h-4 text-primary animate-spin" />
              ) : (
                <Plus className="w-4 h-4 text-muted-foreground" />
              )}
            </div>
            <span className="text-xs text-muted-foreground block">
              {uploadMutation.isPending
                ? uploadProgress !== null
                  ? `Uploading\u2026 ${uploadProgress}%`
                  : "Uploading\u2026"
                : "Drop files or click to add"}
            </span>
            {/* Progress bar */}
            {uploadMutation.isPending && uploadProgress !== null && (
              <div className="w-full h-1 bg-white/10 rounded-full mt-2 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-200 rounded-full"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>

          {uploadError && (
            <div className="mt-2 p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 leading-relaxed">
              <strong>Upload error:</strong> {uploadError}
            </div>
          )}

          <AnimatedButton
            className="w-full h-9 mt-3 text-xs"
            variant="outline"
            onClick={() => setIsPasteModalOpen(true)}
            disabled={!selectedNotebookId || pasteMutation.isPending}
          >
            <FileText className="w-3.5 h-3.5 mr-1.5" />
            {pasteMutation.isPending ? "Ingesting..." : "Paste Text"}
          </AnimatedButton>
        </div>
      </div>

      {/* Center Pane: AI Chat */}
      <div className="flex-1 flex flex-col relative bg-gradient-to-b from-transparent to-surface/20 min-w-0">
        
        {/* Header Controls */}
        <div className="absolute top-0 inset-x-0 p-4 flex justify-between items-center z-20 pointer-events-none">
          <div className="pointer-events-auto flex items-center gap-2">
            <div className="bg-surface/80 backdrop-blur border border-white/5 rounded-full px-3 py-1 text-xs text-muted-foreground flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
              Notebook: <span className="text-foreground font-medium truncate max-w-[120px]">{activeNotebook?.title || "None Selected"}</span>
            </div>
            {selectedDocIds.size > 0 && (
              <div className="bg-primary/15 backdrop-blur border border-primary/30 rounded-full px-3 py-1 text-xs text-primary flex items-center gap-1.5">
                <Filter className="w-3 h-3" />
                {selectedDocIds.size} file{selectedDocIds.size > 1 ? 's' : ''} selected
                <button
                  onClick={() => setSelectedDocIds(new Set())}
                  className="ml-1 hover:text-white transition-colors"
                  title="Clear filter"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}
          </div>
          <div className="pointer-events-auto flex gap-2">
            <AnimatedButton 
              variant="outline" 
              size="sm" 
              className="h-8 bg-surface backdrop-blur-md"
              onClick={handleClearChat}
              disabled={messages.length <= 1}
              title="Clear chat"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Clear Chat
            </AnimatedButton>
            <AnimatedButton 
              variant="outline" 
              size="sm" 
              className={cn("h-8 bg-surface backdrop-blur-md", showGraph && "border-primary text-primary")}
              onClick={() => setShowGraph(!showGraph)}
            >
              <Network className="w-4 h-4 mr-2" />
              Graph View
            </AnimatedButton>
          </div>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto no-scrollbar p-4 md:p-8 pt-16 pb-4 space-y-8 scroll-smooth">
          {messages.map((msg, index) => {
            const isBot = msg.role === "assistant";
            return (
              <div key={index} className={cn("max-w-3xl mx-auto flex gap-4", !isBot && "flex-row-reverse")}>
                {isBot ? (
                  <div className="w-10 h-10 rounded-full flex-shrink-0 bg-primary/20 border border-primary/30 flex items-center justify-center shadow-[0_0_15px_rgba(192,193,255,0.2)]">
                    <Bot className="w-5 h-5 text-primary" />
                  </div>
                ) : (
                  <div className="w-10 h-10 rounded-full flex-shrink-0 border border-white/10 overflow-hidden bg-primary/10 flex items-center justify-center">
                    <span className="text-xs font-mono font-bold text-muted-foreground">U</span>
                  </div>
                )}
                
                {isBot ? (
                  <GlassCard className="rounded-tl-none p-5 max-w-[85%] border-primary/20" hoverEffect={false}>
                    <div className="text-[15px] leading-relaxed text-foreground/90 whitespace-pre-wrap font-sans">
                      {msg.content}
                    </div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <div className="text-xs font-semibold text-foreground mb-2 flex items-center gap-1">
                          <Sparkles className="w-3.5 h-3.5 text-primary" /> Source Citations
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {msg.citations.map((cit, cIdx) => (
                            <div key={cIdx} className="p-3 rounded-lg bg-black/20 border border-white/5 hover:border-primary/30 cursor-pointer transition-colors group">
                              <div className="text-[10px] font-mono text-primary mb-1">[{cIdx + 1}] {cit.file_name}</div>
                              <p className="text-xs text-muted-foreground line-clamp-2 group-hover:text-foreground transition-colors">
                                {cit.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </GlassCard>
                ) : (
                  <div className="bg-surface-hover border border-white/10 rounded-2xl rounded-tr-none p-5 max-w-[85%]">
                    <p className="text-[15px] leading-relaxed text-foreground whitespace-pre-wrap font-sans">
                      {msg.content}
                    </p>
                  </div>
                )}
              </div>
            );
          })}

          {/* Typing Indicator */}
          <AnimatePresence>
            {isTyping && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="max-w-3xl mx-auto flex gap-4"
              >
                <div className="w-10 h-10 rounded-full flex-shrink-0 bg-primary/20 border border-primary/30 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-primary" />
                </div>
                <div className="glass rounded-2xl rounded-tl-none py-4 px-6 flex items-center gap-2 bg-surface/80 border-white/10 border">
                  <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 0.6, repeat: Infinity }} className="w-2 h-2 rounded-full bg-primary/60" />
                  <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 0.6, delay: 0.2, repeat: Infinity }} className="w-2 h-2 rounded-full bg-primary/60" />
                  <motion.div animate={{ y: [0, -5, 0] }} transition={{ duration: 0.6, delay: 0.4, repeat: Infinity }} className="w-2 h-2 rounded-full bg-primary/60" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={chatEndRef} />
        </div>

        {/* Input Area — static, no absolute positioning, no overlap */}
        <div className="flex-shrink-0 border-t border-white/10 bg-background/95 backdrop-blur-md px-4 md:px-8 py-4">
          <div className="max-w-3xl mx-auto flex flex-col gap-2">

            {/* Selected document chips */}
            {selectedDocIds.size > 0 && (
              <div className="flex items-center gap-2 overflow-x-auto no-scrollbar pb-1">
                <span className="text-[10px] text-muted-foreground flex-shrink-0 flex items-center gap-1">
                  <Filter className="w-3 h-3 text-primary" /> Asking only:
                </span>
                {Array.from(selectedDocIds).map(id => {
                  const doc = documents.find(d => d.id === id);
                  if (!doc) return null;
                  return (
                    <span
                      key={id}
                      className="flex-shrink-0 inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-primary/10 border border-primary/25 text-[11px] text-primary font-medium max-w-[160px]"
                    >
                      <FileText className="w-3 h-3 flex-shrink-0" />
                      <span className="truncate">{doc.file_name}</span>
                      <button
                        onClick={() => toggleDocSelection(id)}
                        className="ml-0.5 text-primary/60 hover:text-primary transition-colors flex-shrink-0"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  );
                })}
                <button
                  onClick={() => setSelectedDocIds(new Set())}
                  className="flex-shrink-0 text-[10px] text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-full border border-white/10 hover:border-white/20"
                >
                  Clear all
                </button>
              </div>
            )}

            {/* Quick action suggestion chips */}
            <div className="flex gap-2 overflow-x-auto no-scrollbar">
              <button
                onClick={() => {
                  if (selectedNotebookId && !isTyping) {
                    setChatInput("Summarize the key information contained in this notebook.");
                    textareaRef.current?.focus();
                  }
                }}
                className="whitespace-nowrap px-4 py-1.5 rounded-full bg-surface border border-white/5 text-xs text-muted-foreground hover:text-primary hover:border-primary/30 transition-all cursor-pointer flex-shrink-0"
              >
                Summarize notebook
              </button>
              <button
                onClick={() => {
                  if (selectedNotebookId && !isTyping) {
                    setChatInput("Contrast the main concepts discussed in our documents.");
                    textareaRef.current?.focus();
                  }
                }}
                className="whitespace-nowrap px-4 py-1.5 rounded-full bg-surface border border-white/5 text-xs text-muted-foreground hover:text-accent hover:border-accent/30 transition-all cursor-pointer flex-shrink-0"
              >
                Contrast main concepts
              </button>
            </div>

            {/* Text input box */}
            <div className="glass rounded-[24px] p-2 flex items-end gap-2 border-white/20 shadow-2xl focus-within:border-primary/50 transition-colors bg-surface/80">
              <button
                onClick={() => {
                  if (selectedNotebookId) {
                    fileInputRef.current?.click();
                  } else {
                    alert("Please select a notebook first.");
                  }
                }}
                className="p-3 text-muted-foreground hover:text-foreground transition-colors rounded-xl hover:bg-white/5 mb-0.5 cursor-pointer"
                title="Upload document"
              >
                <Paperclip className="w-5 h-5" />
              </button>

              <textarea
                ref={textareaRef}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder={!selectedNotebookId ? "Please select or create a folder/notebook first..." : selectedDocIds.size > 0 ? `Ask about ${selectedDocIds.size} selected file${selectedDocIds.size > 1 ? 's' : ''}...` : "Ask your AI research assistant (all documents)..."}
                disabled={!selectedNotebookId || isTyping}
                className="flex-1 bg-transparent border-none focus:ring-0 text-[15px] resize-none py-3.5 max-h-[200px] outline-none text-foreground placeholder:text-muted-foreground no-scrollbar"
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
              />

              <div className="flex gap-1 mb-0.5 pr-1">
                <button className="p-2 text-muted-foreground hover:text-foreground transition-colors rounded-xl hover:bg-white/5 hidden sm:block cursor-pointer">
                  <Mic className="w-5 h-5" />
                </button>
                <AnimatedButton
                  variant="primary"
                  size="icon"
                  className="w-10 h-10 rounded-[14px]"
                  onClick={handleSendMessage}
                  disabled={!selectedNotebookId || isTyping || !chatInput.trim()}
                >
                  <Send className="w-4 h-4 ml-0.5" />
                </AnimatedButton>
              </div>
            </div>

            <div className="text-center">
              <span className="text-[10px] text-muted-foreground font-mono">OpenNotebook OS - Llama-3 RAG Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Pane: Graph / Details Panel */}
      <AnimatePresence>
        {showGraph && (
          <motion.div 
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 350, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="flex-shrink-0 border-l border-white/10 bg-surface/30 backdrop-blur-md hidden xl:flex flex-col relative"
          >
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
              <h3 className="font-display font-semibold flex items-center gap-2 text-foreground">
                <Network className="w-4 h-4 text-primary" /> Semantic Graph
              </h3>
              <button onClick={() => setShowGraph(false)} className="text-muted-foreground hover:text-foreground cursor-pointer">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 relative overflow-hidden bg-[#060a14] flex items-center justify-center group cursor-pointer">
              {/* Fake Interactive Graph */}
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(192,193,255,0.1)_0%,transparent_70%)]"></div>
              
              <motion.div animate={{ rotate: 360 }} transition={{ duration: 60, repeat: Infinity, ease: "linear" }} className="relative w-full h-full flex items-center justify-center">
                {/* Central Node */}
                <div className="absolute w-12 h-12 bg-primary/20 border border-primary rounded-full flex items-center justify-center z-20 shadow-[0_0_30px_rgba(192,193,255,0.4)]">
                  <span className="text-xs font-bold text-primary">Wave</span>
                </div>

                {/* Satellite Nodes */}
                <div className="absolute w-8 h-8 bg-accent/20 border border-accent rounded-full flex items-center justify-center translate-x-20 -translate-y-20">
                  <span className="text-[10px] font-bold text-accent">Collapse</span>
                </div>
                <div className="absolute w-6 h-6 bg-secondary/20 border border-secondary rounded-full flex items-center justify-center -translate-x-24 -translate-y-10">
                  <span className="text-[8px] font-bold text-secondary">Bohr</span>
                </div>
                <div className="absolute w-10 h-10 bg-white/10 border border-white/30 rounded-full flex items-center justify-center translate-x-10 translate-y-24">
                  <span className="text-[10px] font-bold text-white/80">Objective</span>
                </div>

                {/* Connecting Lines (SVG) */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ filter: 'drop-shadow(0 0 4px rgba(255,255,255,0.3))' }}>
                  <line x1="50%" y1="50%" x2="calc(50% + 5rem)" y2="calc(50% - 5rem)" stroke="rgba(76,215,246,0.3)" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="calc(50% - 6rem)" y2="calc(50% - 2.5rem)" stroke="rgba(221,183,255,0.3)" strokeWidth="1" />
                  <line x1="50%" y1="50%" x2="calc(50% + 2.5rem)" y2="calc(50% + 6rem)" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4 4" />
                </svg>
              </motion.div>

              <div className="absolute bottom-4 inset-x-0 text-center opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="text-xs font-mono bg-black/50 px-3 py-1 rounded-full border border-white/10 backdrop-blur text-foreground">Click to expand graph</span>
              </div>
            </div>

            <div className="h-1/3 border-t border-white/10 p-4 bg-surface-container overflow-y-auto no-scrollbar">
              <h4 className="text-sm font-bold mb-3 flex items-center gap-2 text-primary">
                <Sparkles className="w-3 h-3" /> Auto-Generated Summary
              </h4>
              <p className="text-xs text-muted-foreground leading-relaxed mb-4">
                {selectedNotebookId && documents.length > 0 
                  ? `Active research folder: "${activeNotebook?.title || "Selected"}". Contains ${documents.length} vectorized document(s). The semantic knowledge graph is fully compiled and ready for conversational exploration.`
                  : "This notebook folder is currently empty. Upload files or scrape websites to start generating interactive semantic connections and visual knowledge graphs."
                }
              </p>
              <AnimatedButton variant="outline" size="sm" className="w-full text-xs">
                Export Analysis
              </AnimatedButton>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Paste Text Modal */}
      <AnimatePresence>
        {isPasteModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-surface border border-white/10 rounded-2xl p-6 w-full max-w-2xl shadow-2xl flex flex-col gap-4 relative"
            >
              <button 
                onClick={() => setIsPasteModalOpen(false)}
                className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                Paste Text
              </h3>
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-muted-foreground">Title (Optional)</label>
                <input
                  type="text"
                  value={pasteTitle}
                  onChange={(e) => setPasteTitle(e.target.value)}
                  placeholder="e.g. Meeting Notes, Web Snippet"
                  className="bg-surface-light border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary/50 text-foreground"
                />
              </div>
              <div className="flex flex-col gap-2 flex-1">
                <label className="text-xs font-medium text-muted-foreground">Content *</label>
                <textarea
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder="Paste your thousands of lines of text here..."
                  className="bg-surface-light border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary/50 text-foreground resize-none h-64"
                />
              </div>
              <div className="flex justify-end gap-3 mt-2">
                <AnimatedButton 
                  variant="outline" 
                  onClick={() => setIsPasteModalOpen(false)}
                >
                  Cancel
                </AnimatedButton>
                <AnimatedButton 
                  variant="primary" 
                  onClick={() => pasteMutation.mutate({ text: pasteText, title: pasteTitle })}
                  disabled={!pasteText.trim() || pasteMutation.isPending}
                >
                  {pasteMutation.isPending ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" /> Ingesting...
                    </span>
                  ) : "Ingest Text"}
                </AnimatedButton>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      
    </div>
  );
}

export default function WorkspacePage() {
  return (
    <Suspense fallback={
      <div className="flex h-screen items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    }>
      <WorkspaceContent />
    </Suspense>
  );
}
