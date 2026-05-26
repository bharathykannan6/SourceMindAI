"use client";

import { useState, useRef, useEffect, Suspense, useCallback } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { GlassCard } from "@/components/ui/GlassCard";
import { AnimatedButton } from "@/components/ui/AnimatedButton";
import { 
  FileText, Search, Folder, Plus, Bot, ArrowUpRight, 
  Paperclip, Mic, Send, Network, Loader2, Sparkles, X, LayoutTemplate, Trash2,
  CheckSquare, CheckCheck, Filter, RotateCcw, Pencil, ExternalLink,
  Download, FileDown
} from "lucide-react";

// Inline YouTube SVG — lucide-react@0.383 does not export Youtube
const YoutubeSVG = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
    <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
  </svg>
);
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
  ingestUrl,
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

// ── MarkdownCitation: renders LLM markdown + hoverable citation badges ──────
//
// Handles the full set of Markdown tokens that Groq/LLaMA produces:
//   ## / ### headers, **bold**, *italic*, `inline code`, ```code blocks```,
//   - / * / + unordered lists, 1. ordered lists, --- horizontal rules,
//   and [N] citation badges injected inline.
//
// Zero external dependencies — pure React + Tailwind.

type MdToken =
  | { t: "h1" | "h2" | "h3"; children: MdToken[] }
  | { t: "p"; children: MdToken[] }
  | { t: "ul"; items: MdToken[][] }
  | { t: "ol"; items: MdToken[][] }
  | { t: "code"; lang: string; text: string }
  | { t: "hr" }
  | { t: "blank" };

type InlineToken =
  | { i: "text"; text: string }
  | { i: "bold"; text: string }
  | { i: "italic"; text: string }
  | { i: "bolditalic"; text: string }
  | { i: "code"; text: string }
  | { i: "cite"; num: number };

// ── Inline tokeniser ─────────────────────────────────────────────────────────
function tokeniseInline(raw: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  // Pattern order matters: longest/most specific first
  const re = /(\*\*\*(.+?)\*\*\*)|(\*\*(.+?)\*\*)|(\*(.+?)\*)|(___(.+?)___)|(__(.+?)__)|(_(. +?)_)|(`([^`]+)`)|(\[(\d+)\])/gs;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) tokens.push({ i: "text", text: raw.slice(last, m.index) });
    if (m[1])  tokens.push({ i: "bolditalic", text: m[2] });
    else if (m[3])  tokens.push({ i: "bold",      text: m[4] });
    else if (m[5])  tokens.push({ i: "italic",    text: m[6] });
    else if (m[7])  tokens.push({ i: "bolditalic", text: m[8] });
    else if (m[9])  tokens.push({ i: "bold",      text: m[10] });
    else if (m[11]) tokens.push({ i: "italic",    text: m[12] });
    else if (m[13]) tokens.push({ i: "code",      text: m[14] });
    else if (m[15]) tokens.push({ i: "cite",      num: parseInt(m[16], 10) });
    last = m.index + m[0].length;
  }
  if (last < raw.length) tokens.push({ i: "text", text: raw.slice(last) });
  return tokens;
}

// ── Block tokeniser ──────────────────────────────────────────────────────────
function tokeniseBlocks(md: string): MdToken[] {
  const lines = md.split("\n");
  const blocks: MdToken[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    const fenceMatch = line.match(/^```(\w*)/);
    if (fenceMatch) {
      const lang = fenceMatch[1] || "";
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      blocks.push({ t: "code", lang, text: codeLines.join("\n") });
      continue;
    }

    // Headings
    if (/^### (.+)/.test(line)) { blocks.push({ t: "h3", children: tokeniseInline(line.replace(/^### /, "")) }); i++; continue; }
    if (/^## (.+)/.test(line))  { blocks.push({ t: "h2", children: tokeniseInline(line.replace(/^## /, "")) }); i++; continue; }
    if (/^# (.+)/.test(line))   { blocks.push({ t: "h1", children: tokeniseInline(line.replace(/^# /, "")) }); i++; continue; }

    // Horizontal rule
    if (/^(---+|\*\*\*+|___+)$/.test(line.trim())) { blocks.push({ t: "hr" }); i++; continue; }

    // Unordered list
    if (/^[-*+] /.test(line)) {
      const items: MdToken[][] = [];
      while (i < lines.length && /^[-*+] /.test(lines[i])) {
        items.push(tokeniseInline(lines[i].replace(/^[-*+] /, "")));
        i++;
      }
      blocks.push({ t: "ul", items });
      continue;
    }

    // Ordered list
    if (/^\d+[.)\s]/.test(line)) {
      const items: MdToken[][] = [];
      while (i < lines.length && /^\d+[.)\s]/.test(lines[i])) {
        items.push(tokeniseInline(lines[i].replace(/^\d+[.)\s]+/, "")));
        i++;
      }
      blocks.push({ t: "ol", items });
      continue;
    }

    // Blank line
    if (line.trim() === "") { blocks.push({ t: "blank" }); i++; continue; }

    // Paragraph — accumulate until blank/special
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,3} |```|[-*+] |\d+[.)\s]|---|\*\*\*|___)/.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length) {
      blocks.push({ t: "p", children: tokeniseInline(paraLines.join(" ")) });
    }
  }
  return blocks;
}

// ── Inline renderer ──────────────────────────────────────────────────────────
function RenderInline({
  tokens,
  citations,
  onCite,
  activeIdx,
}: {
  tokens: InlineToken[];
  citations: Citation[];
  onCite: (num: number, e: React.MouseEvent<HTMLButtonElement>) => void;
  activeIdx: number | null;
}) {
  return (
    <>
      {tokens.map((tok, idx) => {
        switch (tok.i) {
          case "bold":       return <strong key={idx} className="font-semibold text-foreground">{tok.text}</strong>;
          case "italic":     return <em key={idx} className="italic text-foreground/80">{tok.text}</em>;
          case "bolditalic": return <strong key={idx} className="font-semibold italic text-foreground">{tok.text}</strong>;
          case "code":       return <code key={idx} className="px-1.5 py-0.5 rounded bg-white/8 border border-white/10 font-mono text-[13px] text-primary/90">{tok.text}</code>;
          case "cite": {
            const cit = citations[tok.num - 1];
            if (!cit) return <span key={idx} className="text-muted-foreground text-xs">[{tok.num}]</span>;
            return (
              <button
                key={idx}
                onClick={(e) => onCite(tok.num, e)}
                className="inline-flex items-center justify-center w-[18px] h-[18px] mx-[2px] text-[10px] font-bold rounded-full bg-primary/20 text-primary border border-primary/40 hover:bg-primary/40 transition-colors align-super cursor-pointer leading-none shadow-sm"
              >
                {tok.num}
              </button>
            );
          }
          default: return <span key={idx}>{tok.text}</span>;
        }
      })}
    </>
  );
}

// ── Block renderer ───────────────────────────────────────────────────────────
function RenderBlocks({
  blocks,
  citations,
  onCite,
  activeIdx,
}: {
  blocks: MdToken[];
  citations: Citation[];
  onCite: (num: number, e: React.MouseEvent<HTMLButtonElement>) => void;
  activeIdx: number | null;
}) {
  const inlineProps = { citations, onCite, activeIdx };

  return (
    <div className="space-y-3">
      {blocks.map((block, bi) => {
        switch (block.t) {
          case "h1":
            return (
              <h1 key={bi} className="text-xl font-bold text-foreground mt-5 mb-2 leading-snug">
                <RenderInline tokens={block.children} {...inlineProps} />
              </h1>
            );
          case "h2":
            return (
              <h2 key={bi} className="text-[17px] font-semibold text-foreground/95 mt-4 mb-1.5 pb-1 border-b border-white/8 leading-snug">
                <RenderInline tokens={block.children} {...inlineProps} />
              </h2>
            );
          case "h3":
            return (
              <h3 key={bi} className="text-[15px] font-semibold text-primary/90 mt-3 mb-1 leading-snug">
                <RenderInline tokens={block.children} {...inlineProps} />
              </h3>
            );
          case "p":
            return (
              <p key={bi} className="text-[15px] leading-[1.75] text-foreground/90">
                <RenderInline tokens={block.children} {...inlineProps} />
              </p>
            );
          case "ul":
            return (
              <ul key={bi} className="space-y-1.5 pl-1">
                {block.items.map((item, ii) => (
                  <li key={ii} className="flex gap-2.5 text-[15px] leading-[1.7] text-foreground/85">
                    <span className="mt-[7px] w-1.5 h-1.5 rounded-full bg-primary/60 flex-shrink-0" />
                    <span><RenderInline tokens={item} {...inlineProps} /></span>
                  </li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={bi} className="space-y-1.5 pl-1">
                {block.items.map((item, ii) => (
                  <li key={ii} className="flex gap-3 text-[15px] leading-[1.7] text-foreground/85">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 border border-primary/25 text-primary text-[11px] font-bold flex items-center justify-center mt-0.5">{ii + 1}</span>
                    <span className="pt-0.5"><RenderInline tokens={item} {...inlineProps} /></span>
                  </li>
                ))}
              </ol>
            );
          case "code":
            return (
              <div key={bi} className="rounded-xl overflow-hidden border border-white/10 my-2">
                {block.lang && (
                  <div className="px-4 py-1.5 bg-white/5 border-b border-white/10 text-[11px] font-mono text-muted-foreground">{block.lang}</div>
                )}
                <pre className="p-4 overflow-x-auto bg-[#0a0f1e] text-[13px] font-mono text-green-300/90 leading-relaxed">
                  <code>{block.text}</code>
                </pre>
              </div>
            );
          case "hr":
            return <hr key={bi} className="border-white/10 my-3" />;
          case "blank":
            return null;
          default:
            return null;
        }
      })}
    </div>
  );
}

// ── CitationText: top-level component used by the chat ───────────────────────
function CitationText({ content, citations }: { content: string; citations: Citation[] }) {
  const [activeNum, setActiveNum] = useState<number | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ top: number; left: number } | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const blocks = tokeniseBlocks(content);

  useEffect(() => {
    if (activeNum === null) return;
    const handler = (e: MouseEvent) => {
      if (tooltipRef.current?.contains(e.target as Node)) return;
      setActiveNum(null);
      setTooltipPos(null);
    };
    const id = setTimeout(() => document.addEventListener("click", handler), 0);
    return () => { clearTimeout(id); document.removeEventListener("click", handler); };
  }, [activeNum]);

  const handleCite = (num: number, e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    if (activeNum === num) { setActiveNum(null); setTooltipPos(null); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const TW = 300;
    let left = rect.left + rect.width / 2 - TW / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - TW - 8));
    setActiveNum(num);
    setTooltipPos({ top: rect.top - 8, left });
  };

  const activeCit = activeNum !== null ? citations[activeNum - 1] ?? null : null;

  return (
    <>
      <RenderBlocks
        blocks={blocks}
        citations={citations}
        onCite={handleCite}
        activeIdx={activeNum}
      />

      {/* Fixed citation tooltip */}
      {activeNum !== null && activeCit && tooltipPos && (
        <div
          ref={tooltipRef}
          style={{
            position: "fixed",
            top: tooltipPos.top,
            left: tooltipPos.left,
            transform: "translateY(-100%)",
            width: 300,
            zIndex: 9999,
          }}
          className="rounded-2xl border border-primary/30 bg-[#0e1120]/95 shadow-2xl p-4 flex flex-col gap-2 text-left backdrop-blur-md"
        >
          <div className="flex items-center gap-2">
            <span className="w-5 h-5 rounded-full bg-primary/20 border border-primary/40 text-primary text-[10px] font-bold flex items-center justify-center flex-shrink-0">{activeNum}</span>
            <span className="text-[12px] font-semibold text-foreground truncate">{activeCit.file_name}</span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed line-clamp-5">{activeCit.text}</p>
          <button
            onClick={() => { setActiveNum(null); setTooltipPos(null); }}
            className="self-end text-[10px] text-muted-foreground hover:text-foreground mt-1 transition-colors"
          >
            Close
          </button>
        </div>
      )}
    </>
  );
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
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [isPasteModalOpen, setIsPasteModalOpen] = useState(false);
  const [pasteTitle, setPasteTitle] = useState("");
  const [pasteText, setPasteText] = useState("");

  const [isYoutubeModalOpen, setIsYoutubeModalOpen] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeFetchingTitle, setYoutubeFetchingTitle] = useState(false);
  const [youtubeTitle, setYoutubeTitle] = useState("");
  const [youtubeError, setYoutubeError] = useState<string | null>(null);
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

  // YouTube Ingest Mutation
  const youtubeMutation = useMutation({
    mutationFn: ({ url, title }: { url: string; title: string }) =>
      ingestUrl(url, title || undefined, selectedNotebookId || undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
      setIsYoutubeModalOpen(false);
      setYoutubeUrl("");
      setYoutubeTitle("");
      setYoutubeError(null);
    },
    onError: (error: any) => {
      const msg = error?.response?.data?.detail || error?.message || "Failed to fetch transcript.";
      setYoutubeError(msg);
    }
  });

  // Fetch real video title from YouTube oEmbed when URL changes
  const handleYoutubeUrlChange = async (url: string) => {
    setYoutubeUrl(url);
    setYoutubeError(null);
    const isYT = url.includes("youtube.com/watch") || url.includes("youtu.be/");
    if (!isYT) { setYoutubeTitle(""); return; }
    setYoutubeFetchingTitle(true);
    try {
      const oembedUrl = `https://www.youtube.com/oembed?url=${encodeURIComponent(url)}&format=json`;
      const res = await fetch(oembedUrl);
      if (res.ok) {
        const data = await res.json();
        setYoutubeTitle(data.title || "");
      }
    } catch { /* silently ignore */ }
    finally { setYoutubeFetchingTitle(false); }
  };

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

  // Reset doc selection and conversation when notebook changes
  useEffect(() => {
    setSelectedDocIds(new Set());
    setConversationId(null);
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
        selectedDocIds.size > 0 ? Array.from(selectedDocIds) : undefined,
        conversationId || undefined
      );
      // Persist conversation_id for memory continuity
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
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
    setConversationId(null);
    setMessages([
      {
        role: "assistant",
        content: "I've fully ingested your workspace documents. The vectors are mapped and the knowledge graph is ready. How would you like to explore your notes and sources today?",
      }
    ]);
  };

  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const activeNotebook = notebooks.find(n => n.id === selectedNotebookId);

  const exportMenuRef = useRef<HTMLDivElement>(null);

  // Close export menu on outside click
  useEffect(() => {
    if (!isExportMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setIsExportMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [isExportMenuOpen]);

  // Strip markdown to plain text for export
  const stripMarkdown = (md: string) => md
    .replace(/#{1,6}\s+/g, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/`{1,3}[^`]*`{1,3}/g, "")
    .replace(/\[\d+\]/g, "")
    .replace(/^[-*+]\s+/gm, "\u2022 ")
    .replace(/^\d+\.\s+/gm, "")
    .trim();

  const exportAsPDF = useCallback(async () => {
    if (messages.length <= 1) return;
    setIsExporting(true);
    setIsExportMenuOpen(false);
    try {
      const notebookTitle = activeNotebook?.title || "OpenNotebook Export";
      const exportDate = new Date().toLocaleDateString("en-IN", { year: "numeric", month: "long", day: "numeric" });

      // Build HTML for PDF
      const chatHtml = messages.map((msg) => {
        const isBot = msg.role === "assistant";
        const text = stripMarkdown(msg.content)
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br/>");
        return `
          <div style="margin-bottom:20px;">
            <div style="font-size:11px;font-weight:600;color:${isBot ? "#6366f1" : "#888"};margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;">
              ${isBot ? "OpenNotebook AI" : "You"}
            </div>
            <div style="background:${isBot ? "#f0f0ff" : "#f9f9f9"};border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.7;color:#1a1a2e;border-left:3px solid ${isBot ? "#6366f1" : "#ddd"};">${text}</div>
          </div>`;
      }).join("");

      const html = `<!DOCTYPE html><html><head><meta charset="utf-8">
        <title>${notebookTitle} — Chat Export</title>
        <style>
          body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;padding:0 24px;color:#1a1a2e;}
          h1{font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px;}
          .meta{font-size:12px;color:#888;margin-bottom:32px;}
          hr{border:none;border-top:1px solid #e5e5e5;margin:24px 0;}
        </style>
      </head><body>
        <h1>${notebookTitle} — Chat Export</h1>
        <div class="meta">Exported from OpenNotebook · ${exportDate}</div>
        <hr/>${chatHtml}
      </body></html>`;

      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const win = window.open(url, "_blank");
      if (win) {
        win.onload = () => {
          win.print();
          setTimeout(() => URL.revokeObjectURL(url), 60000);
        };
      }
    } finally {
      setIsExporting(false);
    }
  }, [messages, activeNotebook]);

  const exportAsTXT = useCallback(() => {
    if (messages.length <= 1) return;
    setIsExportMenuOpen(false);
    const notebookTitle = activeNotebook?.title || "OpenNotebook Export";
    const exportDate = new Date().toLocaleDateString("en-IN");
    const lines: string[] = [
      `${notebookTitle} — Chat Export`,
      `Exported: ${exportDate}`,
      `${"=".repeat(60)}`,
      "",
    ];
    messages.forEach((msg) => {
      lines.push(msg.role === "assistant" ? "[OpenNotebook AI]" : "[You]");
      lines.push(stripMarkdown(msg.content));
      lines.push("");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${notebookTitle.replace(/\s+/g, "_")}_export.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [messages, activeNotebook]);

  const exportAsMarkdown = useCallback(() => {
    if (messages.length <= 1) return;
    setIsExportMenuOpen(false);
    const notebookTitle = activeNotebook?.title || "OpenNotebook Export";
    const exportDate = new Date().toLocaleDateString("en-IN");
    const lines: string[] = [
      `# ${notebookTitle} — Chat Export`,
      `> Exported from OpenNotebook · ${exportDate}`,
      "",
      "---",
      "",
    ];
    messages.forEach((msg) => {
      lines.push(msg.role === "assistant" ? "### 🤖 OpenNotebook AI" : "### 👤 You");
      lines.push("");
      lines.push(msg.content);
      lines.push("");
      lines.push("---");
      lines.push("");
    });
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${notebookTitle.replace(/\s+/g, "_")}_export.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [messages, activeNotebook]);

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

          <div className="flex gap-2 mt-3">
            <AnimatedButton
              className="flex-1 h-9 text-xs"
              variant="outline"
              onClick={() => setIsPasteModalOpen(true)}
              disabled={!selectedNotebookId || pasteMutation.isPending}
            >
              <FileText className="w-3.5 h-3.5 mr-1.5" />
              {pasteMutation.isPending ? "Ingesting..." : "Paste Text"}
            </AnimatedButton>
            <AnimatedButton
              className="flex-1 h-9 text-xs"
              variant="outline"
              onClick={() => {
                if (!selectedNotebookId) { alert("Please select a notebook first."); return; }
                setYoutubeUrl("");
                setYoutubeTitle("");
                setYoutubeError(null);
                setIsYoutubeModalOpen(true);
              }}
              disabled={!selectedNotebookId || youtubeMutation.isPending}
            >
              <YoutubeSVG className="w-3.5 h-3.5 mr-1.5 text-red-400" />
              {youtubeMutation.isPending ? "Fetching..." : "YouTube"}
            </AnimatedButton>
          </div>
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
            {/* Export Menu */}
            <div className="relative" ref={exportMenuRef}>
              <AnimatedButton
                variant="outline"
                size="sm"
                className="h-8 bg-surface backdrop-blur-md"
                onClick={() => setIsExportMenuOpen(prev => !prev)}
                disabled={messages.length <= 1 || isExporting}
                title="Export chat"
              >
                {isExporting ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <FileDown className="w-4 h-4 mr-2" />
                )}
                Export
              </AnimatedButton>

              <AnimatePresence>
                {isExportMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -6, scale: 0.96 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.96 }}
                    transition={{ duration: 0.12 }}
                    className="absolute right-0 top-10 z-50 w-44 bg-surface border border-white/10 rounded-xl shadow-2xl overflow-hidden"
                  >
                    <button
                      onClick={exportAsPDF}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-white/5 transition-colors text-left"
                    >
                      <FileText className="w-4 h-4 text-red-400" />
                      Export as PDF
                    </button>
                    <button
                      onClick={exportAsMarkdown}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-white/5 transition-colors text-left border-t border-white/5"
                    >
                      <FileDown className="w-4 h-4 text-blue-400" />
                      Export as .md
                    </button>
                    <button
                      onClick={exportAsTXT}
                      className="w-full flex items-center gap-3 px-4 py-3 text-sm text-foreground hover:bg-white/5 transition-colors text-left border-t border-white/5"
                    >
                      <Download className="w-4 h-4 text-green-400" />
                      Export as .txt
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
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
                    <div className="text-[15px] leading-relaxed text-foreground/90 font-sans">
                      <CitationText content={msg.content} citations={msg.citations || []} />
                    </div>
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
      {/* YouTube URL Modal */}
      <AnimatePresence>
        {isYoutubeModalOpen && (
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
              className="bg-surface border border-white/10 rounded-2xl p-6 w-full max-w-lg shadow-2xl flex flex-col gap-4 relative"
            >
              <button
                onClick={() => setIsYoutubeModalOpen(false)}
                className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="w-5 h-5" />
              </button>

              <h3 className="text-lg font-semibold flex items-center gap-2">
                <YoutubeSVG className="w-5 h-5 text-red-400" />
                Add YouTube Video
              </h3>

              <p className="text-xs text-muted-foreground -mt-2">
                Paste a YouTube link. The transcript will be fetched and stored as a document you can chat with.
              </p>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-muted-foreground">YouTube URL *</label>
                <div className="relative">
                  <input
                    type="url"
                    value={youtubeUrl}
                    onChange={(e) => handleYoutubeUrlChange(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=..."
                    className="w-full bg-surface-light border border-white/10 rounded-lg px-3 py-2 pr-10 text-sm focus:outline-none focus:border-primary/50 text-foreground"
                    autoFocus
                  />
                  {youtubeFetchingTitle && (
                    <Loader2 className="w-4 h-4 animate-spin text-primary absolute right-3 top-2.5" />
                  )}
                </div>
              </div>

              {/* Auto-fetched title preview */}
              {youtubeTitle && (
                <div className="flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-lg px-3 py-2">
                  <YoutubeSVG className="w-4 h-4 text-red-400 flex-shrink-0" />
                  <span className="text-sm text-foreground truncate">{youtubeTitle}</span>
                </div>
              )}

              {/* Validation message */}
              {youtubeUrl && !youtubeUrl.includes("youtube.com") && !youtubeUrl.includes("youtu.be") && (
                <p className="text-xs text-yellow-400 flex items-center gap-1">
                  <ExternalLink className="w-3 h-3" /> This doesn&apos;t look like a YouTube URL.
                </p>
              )}

              {/* Error */}
              {youtubeError && (
                <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400">
                  <strong>Error:</strong> {youtubeError}
                  {youtubeError.includes("Transcript") || youtubeError.includes("transcript") ? (
                    <p className="mt-1 text-red-300">This video may have transcripts disabled or be in a language not supported.</p>
                  ) : null}
                </div>
              )}

              <div className="flex justify-end gap-3 mt-2">
                <AnimatedButton variant="outline" onClick={() => setIsYoutubeModalOpen(false)}>
                  Cancel
                </AnimatedButton>
                <AnimatedButton
                  variant="primary"
                  onClick={() => youtubeMutation.mutate({ url: youtubeUrl, title: youtubeTitle })}
                  disabled={
                    !youtubeUrl.trim() ||
                    (!youtubeUrl.includes("youtube.com") && !youtubeUrl.includes("youtu.be")) ||
                    youtubeMutation.isPending
                  }
                >
                  {youtubeMutation.isPending ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" /> Fetching transcript...
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <YoutubeSVG className="w-4 h-4" /> Add Transcript
                    </span>
                  )}
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
