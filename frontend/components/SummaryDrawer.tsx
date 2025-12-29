"use client";

import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cx } from "@/lib/cx";
import { SummaryItem } from "@/types/chat";

type SummaryDrawerProps = {
  isOpen: boolean;
  summaries: SummaryItem[];
  selectedSummary: SummaryItem | null;
  onSelectSummary: (id: string | null) => void;
  onDeleteSummary: (id: string) => void;
  onClose: () => void;
  formatSummaryStamp: (ts: number) => string;
  stripSummaryTitle: (text: string) => string;
};

export function SummaryDrawer({
  isOpen,
  summaries,
  selectedSummary,
  onSelectSummary,
  onDeleteSummary,
  onClose,
  formatSummaryStamp,
  stripSummaryTitle,
}: SummaryDrawerProps) {
  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-30 bg-transparent backdrop-blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={onClose}
              aria-hidden="true"
            />
            <motion.aside
              className="fixed left-0 top-0 z-40 h-full w-96 max-w-[90vw] overflow-y-auto bg-transparent px-6 pb-6 pt-5"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", stiffness: 220, damping: 26 }}
              role="dialog"
              aria-label="Sidebar"
            >
              <div className="flex items-center justify-between">
                <div className="text-xs uppercase tracking-[0.4em] text-slate-400">
                  Summaries
                </div>
              </div>
              <div className="mt-6 flex h-[calc(100%-3.5rem)] gap-4 text-sm text-slate-300">
                <div className="flex w-full flex-col gap-3">
                  <div className="flex-1 space-y-3 overflow-y-auto pr-1">
                    {summaries.length === 0 ? (
                      <div className="rounded-2xl glass-panel glass-panel--input px-4 py-4 text-xs text-slate-400">
                        No summaries yet. Type /summary to create one.
                      </div>
                    ) : (
                      summaries.map((s) => (
                        <div
                          key={s.id}
                          onClick={() => onSelectSummary(s.id)}
                          className={cx(
                            "group relative w-full rounded-2xl border px-4 py-3 text-left transition glass-panel glass-panel--input"
                          )}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              onSelectSummary(s.id);
                            }
                          }}
                        >
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteSummary(s.id);
                            }}
                            className="absolute right-3 top-3 inline-flex h-6 w-6 items-center justify-center rounded-md border border-slate-800/60 text-slate-400 opacity-0 transition group-hover:opacity-100 hover:text-slate-200"
                            title="Delete summary"
                          >
                            <svg
                              viewBox="0 0 24 24"
                              className="h-3.5 w-3.5"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="1.8"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              aria-hidden="true"
                            >
                              <path d="M3 6h18" />
                              <path d="M8 6V4h8v2" />
                              <path d="M6 6l1 14h10l1-14" />
                            </svg>
                          </button>
                          <div className="text-[10px] uppercase tracking-[0.28em] text-slate-400">
                            {formatSummaryStamp(s.createdAt)}
                          </div>
                          <div className="mt-1 text-sm font-semibold text-slate-100 truncate">
                            {s.title}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>

                <div className="flex min-w-0 flex-1 flex-col" />
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isOpen && selectedSummary ? (
          <motion.section
            className="fixed top-6 right-6 left-6 lg:left-100 z-40 h-[calc(100%-3rem)] rounded-3xl border border-slate-800/60 bg-slate-950/85 p-5 shadow-[0_0_40px_rgba(12,255,220,0.08)] glass-panel glass-panel--input"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 12 }}
            transition={{ duration: 0.2 }}
            role="dialog"
            aria-label="Summary detail"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="mt-1 text-sm font-semibold text-slate-100">
                  {selectedSummary.title}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => onDeleteSummary(selectedSummary.id)}
                  className="inline-flex h-8 items-center gap-2 rounded-lg border border-slate-800/60 px-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-300 transition hover:text-slate-100"
                  title="Delete summary"
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M3 6h18" />
                    <path d="M8 6V4h8v2" />
                    <path d="M6 6l1 14h10l1-14" />
                  </svg>
                  Delete
                </button>
                <button
                  type="button"
                  onClick={() => onSelectSummary(null)}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-800/60 text-slate-300 transition hover:text-slate-100"
                  title="Close summary"
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M6 6l12 12" />
                    <path d="M18 6l-12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="mt-4 h-[calc(100%-3.5rem)] overflow-y-auto rounded-2xl border border-slate-800/50 bg-black/30 p-4">
              {selectedSummary.error ? (
                <div className="text-xs text-rose-300">{selectedSummary.error}</div>
              ) : selectedSummary.isLoading && !selectedSummary.content ? (
                <div className="text-xs text-slate-400">Generating summary...</div>
              ) : (
                <div className="text-[14px] leading-relaxed text-slate-100/90">
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
                        return (
                          <pre className="my-3 overflow-x-auto rounded-xl border border-white/10 bg-black/50 p-3 text-xs font-mono">
                            <code className={className} {...props}>
                              {children}
                            </code>
                          </pre>
                        );
                      },
                    }}
                  >
                    {stripSummaryTitle(selectedSummary.content)}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </motion.section>
        ) : null}
      </AnimatePresence>
    </>
  );
}
