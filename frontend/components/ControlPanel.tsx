"use client";

import { DocsMenu } from "@/components/DocsMenu";
import { IndexStatusCard } from "@/components/IndexStatusCard";
import { MetricsCard } from "@/components/MetricsCard";
import { DocItem, IndexStatus, MetricsResponse } from "@/types/chat";

type ControlPanelProps = {
  docs: DocItem[];
  docsLabel: string;
  onUpload: (file: File) => void;
  uploading: boolean;
  indexStatus: IndexStatus | null;
  indexProgress: number | null;
  indexStarting: boolean;
  onStartIndex: () => void;
  metrics: MetricsResponse | null;
  formatBytes: (bytes?: number | null) => string;
  modelFallback: string;
};

export function ControlPanel({
  docs,
  docsLabel,
  onUpload,
  uploading,
  indexStatus,
  indexProgress,
  indexStarting,
  onStartIndex,
  metrics,
  formatBytes,
  modelFallback,
}: ControlPanelProps) {
  return (
    <div className="absolute top-4 right-4 z-10 isolate">
      <div className="inline-flex flex-col items-stretch gap-2 bg-transparent">
        <div className="mt-6 inline-flex items-center gap-2 rounded-2xl px-2.5 py-1.5 text-[11px] text-slate-300 glass-panel glass-panel--thin neon-edge">
          <DocsMenu docs={docs} label={docsLabel} />
          <label
            className="rounded-lg border border-slate-800/70 bg-transparent px-2 py-1 text-[11px] font-semibold text-slate-200 transition hover:border-emerald-300/60 hover:text-emerald-100"
            title="Upload a document into data/documents and auto-index"
          >
            <span className="inline-flex items-center gap-1.5">
              <svg
                className="h-3.5 w-3.5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 3v12" />
                <path d="M7 8l5-5 5 5" />
                <path d="M5 21h14" />
              </svg>
              <span>Upload</span>
            </span>
            <input
              type="file"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
                e.currentTarget.value = "";
              }}
              disabled={uploading}
            />
          </label>
        </div>

        <IndexStatusCard
          indexStatus={indexStatus}
          indexProgress={indexProgress}
          indexStarting={indexStarting}
          onStartIndex={onStartIndex}
        />

        <MetricsCard
          metrics={metrics}
          formatBytes={formatBytes}
          modelFallback={modelFallback}
        />
      </div>
    </div>
  );
}
