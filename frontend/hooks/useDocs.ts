"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/constants/api";
import { DocItem } from "@/types/chat";

export function useDocs() {
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/docs/list`);
      const data = await r.json();
      setDocs(data.docs || []);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  return { docs, loading, refreshDocs: fetchDocs, setDocs };
}
