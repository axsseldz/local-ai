"use client";

import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cx } from "@/lib/cx";
import { ChatMessage } from "@/types/chat";

type Props = {
  messages: ChatMessage[];
  formatAssistantContent: (message: ChatMessage) => string;
};

export function MessageList({ messages, formatAssistantContent }: Props) {
  return (
    <div className="space-y-3">
      <AnimatePresence>
        {messages.map((m) => {
          const formatted = m.role === "assistant" ? formatAssistantContent(m) : m.content;
          return (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className={cx(
                "rounded-3xl overflow-hidden message-shell",
                m.role === "user"
                  ? "ml-auto message-user"
                  : "mr-auto message-ai"
              )}
            >
              {m.role === "assistant" ? (
                <div className="p-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="font-semibold">Jarvis</div>
                    {m.mode && (
                      <span className="text-xs text-slate-400">
                        {m.mode}
                      </span>
                    )}
                  </div>
                </div>
              ) : null}

              <div className={cx("px-5 pb-5", m.role === "user" && "pt-4 text-left")}>
                {m.role === "assistant" ? (
                  m.isLoading ? (
                    <div className="flex items-center gap-3 text-sm text-slate-300">
                      <div className="loading-dots" aria-hidden="true">
                        <span className="loading-dot" />
                        <span className="loading-dot" />
                        <span className="loading-dot" />
                      </div>
                      <span className="text-[11px] uppercase tracking-[0.3em] text-emerald-200/70">
                        Loading
                      </span>
                    </div>
                  ) : (
                    <div className="max-w-none text-[15px] leading-relaxed text-slate-100/95">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          table({ children, ...props }) {
                            return (
                              <div className="my-4 w-full overflow-x-auto rounded-xl border border-white/10">
                                <table className="w-full border-collapse text-sm" {...props}>
                                  {children}
                                </table>
                              </div>
                            );
                          },
                          thead({ children, ...props }) {
                            return (
                              <thead className="bg-white/5" {...props}>
                                {children}
                              </thead>
                            );
                          },
                          th({ children, ...props }) {
                            return (
                              <th className="px-3 py-2 text-left font-semibold border-b border-white/10" {...props}>
                                {children}
                              </th>
                            );
                          },
                          td({ children, ...props }) {
                            return (
                              <td className="px-3 py-2 align-top border-b border-white/5" {...props}>
                                {children}
                              </td>
                            );
                          },
                          code({ className, children, ...props }) {
                            const isBlock = typeof className === "string" && className.includes("language-");
                            if (!isBlock) {
                              return (
                                <code className="rounded-md bg-white/5 px-1.5 py-0.5 text-[0.95em] font-mono" {...props}>
                                  {children}
                                </code>
                              );
                            }
                            const codeText = Array.isArray(children) ? children.join("") : String(children ?? "");
                            return (
                              <div className="my-4 rounded-xl border border-white/10 bg-black/40">
                                <div className="relative">
                                  <button
                                    type="button"
                                    className="absolute right-2 top-2 inline-flex items-center gap-1.5 rounded-md border border-cyan-400/40 bg-linear-to-r from-cyan-500/15 via-sky-500/10 to-teal-500/15 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-100/90 shadow-[0_0_12px_rgba(34,211,238,0.15)] transition hover:border-cyan-300/70 hover:text-cyan-100 hover:shadow-[0_0_14px_rgba(34,211,238,0.25)]"
                                    onClick={(e) => {
                                      const btn = e.currentTarget;
                                      if (!navigator?.clipboard) return;
                                      navigator.clipboard.writeText(codeText).then(() => {
                                        btn.setAttribute("data-state", "copied");
                                        btn.textContent = "Copied";
                                        window.setTimeout(() => {
                                          btn.removeAttribute("data-state");
                                          btn.textContent = "Copy";
                                        }, 1200);
                                      });
                                    }}
                                  >
                                    <svg
                                      className="h-3 w-3"
                                      viewBox="0 0 24 24"
                                      fill="none"
                                      stroke="currentColor"
                                      strokeWidth="1.6"
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      aria-hidden="true"
                                    >
                                      <rect x="9" y="9" width="12" height="12" rx="2" />
                                      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
                                    </svg>
                                    Copy
                                  </button>
                                  <pre className="overflow-x-auto rounded-xl p-4 text-sm font-mono">
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  </pre>
                                </div>
                              </div>
                            );
                          },
                        }}
                      >
                        {formatted}
                      </ReactMarkdown>

                    </div>
                  )
                ) : (
                  <div className="whitespace-pre-wrap text-sm text-slate-100 leading-relaxed">
                    {m.content}
                  </div>
                )}

                {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                  <div className="mt-5">
                    <details className="group">
                      <summary className="cursor-pointer list-none flex items-center justify-between rounded-2xl border border-slate-800/60 bg-black/50 px-4 py-3 hover:bg-black/70 transition">
                        <span className="text-sm font-semibold">
                          Sources ({m.sources.length})
                        </span>
                        <span className="text-xs text-zinc-400 group-open:rotate-180 transition-transform">
                          ▾
                        </span>
                      </summary>

                      <div className="mt-3 grid gap-2">
                        {m.sources.map((s) => {
                          const isWeb = !!s.url;

                          return (
                            <div key={s.label} className="rounded-xl border border-slate-700/50 p-3 glass-panel glass-panel--thin">
                              <div className="text-xs font-semibold text-slate-300/70">
                                {s.label} {isWeb ? "· Web" : "· Local"}
                              </div>

                              {isWeb ? (
                                <div className="mt-1">
                                  <a
                                    href={s.url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-sm font-medium text-sky-300 hover:underline"
                                  >
                                    {s.title || s.url}
                                  </a>
                                  {s.snippet ? (
                                    <div className="mt-1 text-sm text-slate-300/80">
                                      {s.snippet}
                                    </div>
                                  ) : null}
                                </div>
                              ) : (
                                <div className="mt-1 text-sm text-slate-300/80">
                                  <div className="truncate">{s.doc_path}</div>
                                  <div className="text-xs text-slate-400/80">
                                    chunk {s.chunk_index} {typeof s.score === "number" ? `· score ${s.score.toFixed(3)}` : ""}
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}

                      </div>
                    </details>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
