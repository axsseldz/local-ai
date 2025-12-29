"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/constants/api";
import { IndexStatus } from "@/types/chat";

export function useIndexStatus() {
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexStarting, setIndexStarting] = useState(false);

  const fetchIndexStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/index/status`);
      const data = (await r.json()) as IndexStatus;
      setIndexStatus(data);
    } catch {
      // ignore
    }
  }, []);

  const startIndexNow = useCallback(async () => {
    try {
      setIndexStarting(true);
      await fetch(`${API_BASE}/index/run`, { method: "POST" });
      setTimeout(fetchIndexStatus, 400);
    } finally {
      setIndexStarting(false);
    }
  }, [fetchIndexStatus]);

  useEffect(() => {
    fetchIndexStatus();

    const t = setInterval(() => {
      if (indexStatus?.is_indexing) fetchIndexStatus();
    }, 1200);

    const slow = setInterval(() => {
      if (!indexStatus?.is_indexing) fetchIndexStatus();
    }, 8000);

    return () => {
      clearInterval(t);
      clearInterval(slow);
    };
  }, [fetchIndexStatus, indexStatus?.is_indexing]);

  return { indexStatus, indexStarting, fetchIndexStatus, startIndexNow, setIndexStatus };
}
