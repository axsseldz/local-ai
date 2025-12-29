export type Mode = "local" | "general" | "search";

export type Source = {
  label: string;
  doc_path?: string;
  chunk_index?: number;
  score?: number | null;
  title?: string;
  url?: string;
  snippet?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  mode?: string;
  task?: string;
  model?: string;
  sources?: Source[];
  error?: boolean;
  isLoading?: boolean;
};

export type SummaryItem = {
  id: string;
  title: string;
  preview: string;
  content: string;
  createdAt: number;
  isLoading?: boolean;
  error?: string;
};

export type IndexStatus = {
  state: "idle" | "running" | "ok" | "error";
  is_indexing: boolean;
  last_trigger?: string | null;
  last_started_at?: string | null;
  last_finished_at?: string | null;
  last_error?: string | null;
  stats?: {
    files_on_disk: number;
    indexed_files: number;
    skipped_files: number;
    deleted_files: number;
    chunks_indexed: number;
  } | null;
};

export type DocItem = {
  name: string;
  path: string;
  size: number;
  modified_at: string;
};

export type MetricsResponse = {
  system?: {
    memory_used_bytes?: number | null;
    memory_total_bytes?: number | null;
    memory_free_bytes?: number | null;
    memory_active_bytes?: number | null;
    memory_wired_bytes?: number | null;
    memory_compressed_bytes?: number | null;
    memory_inactive_bytes?: number | null;
    memory_speculative_bytes?: number | null;
    memory_purgeable_bytes?: number | null;
    swap_used_bytes?: number | null;
    swap_total_bytes?: number | null;
    cpu_percent?: number | null;
    gpu_util_percent?: number | null;
    metal_supported?: boolean | null;
  };
  llm?: {
    tokens_per_second?: number | null;
    ttft_ms?: number | null;
    context_chars?: number | null;
    last_updated?: number | null;
    rss_bytes?: number | null;
  };
  model?: {
    name?: string | null;
    quantization?: string | null;
    backend?: string | null;
  };
};
