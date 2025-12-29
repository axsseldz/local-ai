"use client";

import { cx } from "@/lib/cx";
import { MetricsResponse } from "@/types/chat";

type Props = {
  metrics: MetricsResponse | null;
  formatBytes: (bytes?: number | null) => string;
  modelFallback: string | null;
};

export function MetricsCard({ metrics, formatBytes, modelFallback }: Props) {
  const memUsed = metrics?.system?.memory_used_bytes ?? null;
  const memTotal = metrics?.system?.memory_total_bytes ?? null;
  const memFree = metrics?.system?.memory_free_bytes ?? null;
  const memActive = metrics?.system?.memory_active_bytes ?? null;
  const memWired = metrics?.system?.memory_wired_bytes ?? null;
  const memCompressed = metrics?.system?.memory_compressed_bytes ?? null;
  const swapUsed = metrics?.system?.swap_used_bytes ?? null;
  const cpuPercent = metrics?.system?.cpu_percent ?? null;
  const gpuUtil = metrics?.system?.gpu_util_percent ?? null;
  const metalSupported = metrics?.system?.metal_supported ?? null;
  const tps = metrics?.llm?.tokens_per_second ?? null;
  const ttft = metrics?.llm?.ttft_ms ?? null;
  const contextChars = metrics?.llm?.context_chars ?? null;
  const modelName = metrics?.model?.name ?? modelFallback ?? null;
  const modelQuant = metrics?.model?.quantization ?? null;
  const modelBackend = metrics?.model?.backend ?? null;

  const memRatio = memUsed && memTotal ? memUsed / memTotal : null;
  const memPressure =
    memRatio == null ? "unknown" : memRatio < 0.75 ? "green" : memRatio < 0.9 ? "yellow" : "red";

  return (
    <div className="relative z-0 w-full rounded-2xl px-3 py-2 glass-panel neon-edge">
      <div className="flex items-center justify-between gap-3 text-[11px] font-semibold text-slate-200">
        <span>LLM Resource Monitor</span>
        <span className="rounded-full bg-slate-700/30 px-2 py-0.5 text-[10px] text-slate-200">
          Live
        </span>
      </div>

      <div className="mt-2 space-y-2 text-[11px] text-slate-300">
        <div className="flex items-center justify-between">
          <span>RAM used</span>
          <span className="text-slate-100">
            {formatBytes(memUsed)} / {formatBytes(memTotal)}
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900/60">
          <div
            className={cx(
              "h-full transition-[width]",
              memPressure === "green" && "bg-emerald-400/70",
              memPressure === "yellow" && "bg-amber-400/70",
              memPressure === "red" && "bg-rose-400/70",
              memPressure === "unknown" && "bg-slate-600/40"
            )}
            style={{ width: memRatio ? `${Math.round(memRatio * 100)}%` : "10%" }}
          />
        </div>

        <div className="flex items-center justify-between">
          <span>LLM RAM</span>
          <span className="text-slate-100">{formatBytes(metrics?.llm?.rss_bytes ?? null)}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-400">
          <div className="flex items-center justify-between">
            <span>Free</span>
            <span className="text-slate-200">{formatBytes(memFree)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Active</span>
            <span className="text-slate-200">{formatBytes(memActive)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Wired</span>
            <span className="text-slate-200">{formatBytes(memWired)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Compressed</span>
            <span className="text-slate-200">{formatBytes(memCompressed)}</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span>Swap</span>
          <span className={swapUsed ? "text-rose-200" : "text-emerald-200"}>
            {swapUsed == null ? "—" : `${formatBytes(swapUsed)}${swapUsed ? " in use" : " ok"}`}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center justify-between">
            <span>CPU</span>
            <span className="text-slate-100">
              {cpuPercent != null
                ? `${cpuPercent.toFixed(1)}%`
                : "—"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>GPU</span>
            <span className="text-slate-100">
              {gpuUtil != null
                ? `${gpuUtil.toFixed(0)}%`
                : "—"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Metal</span>
            <span
              className={cx(
                "text-slate-100",
                metalSupported === false && "text-rose-200",
                metalSupported === true && "text-emerald-200"
              )}
            >
              {metalSupported == null
                ? "Unknown"
                : metalSupported
                  ? "Active"
                  : "Off"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>Context</span>
            <span className="text-slate-100">
              {contextChars != null ? `${contextChars}` : "—"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="flex items-center justify-between">
            <span>TPS</span>
            <span className="text-slate-100">
              {tps != null ? tps : "—"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span>TTFT</span>
            <span className="text-slate-100">
              {ttft != null ? `${ttft} ms` : "—"}
            </span>
          </div>
        </div>

        <div className="pt-1 text-[10px] text-slate-400">
          Model {modelName ?? "—"} · Quant {modelQuant ?? "—"} · {modelBackend ?? "—"}
        </div>
      </div>
    </div>
  );
}
