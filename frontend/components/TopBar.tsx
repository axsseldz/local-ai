"use client";

import { ModelMenu } from "@/components/ModelMenu";

type TopBarProps = {
  onOpenSidebar: () => void;
  model: string;
  modelOptions: string[];
  onModelChange: (m: string) => void;
};

export function TopBar({ onOpenSidebar, model, modelOptions, onModelChange }: TopBarProps) {
  return (
    <div className="absolute top-4 left-4 z-20">
      <div className="mt-6 flex items-center gap-3">
        <button
          type="button"
          onClick={onOpenSidebar}
          className="inline-flex h-11 w-11 mb-1 items-center justify-center rounded-2xl border-0"
          aria-label="Open sidebar"
        >
          <svg
            viewBox="0 0 24 24"
            className="h-7 w-7"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M4 6h16" />
            <path d="M4 12h16" />
            <path d="M4 18h16" />
          </svg>
        </button>
        <ModelMenu model={model} modelOptions={modelOptions} onChange={onModelChange} />
      </div>
    </div>
  );
}
