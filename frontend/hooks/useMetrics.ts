"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/constants/api";
import { MetricsResponse } from "@/types/chat";

export function useMetrics() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (!res.ok) return;
      const data = (await res.json()) as MetricsResponse;
      setMetrics(data);
    } catch {
      setMetrics(null);
    }
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      await fetchMetrics();
    };

    tick();
    const id = setInterval(tick, 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [fetchMetrics]);

  return { metrics, refreshMetrics: fetchMetrics };
}
