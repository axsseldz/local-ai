export function stripForSpeech(input: string) {
  let t = input || "";
  t = t.replace(/```[\s\S]*?```/g, " ");
  t = t.replace(/`([^`]+)`/g, "$1");
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  t = t.replace(/\[S\d+\]/g, "");
  t = t.replace(/\s+/g, " ").trim();
  if (/^loading/i.test(t)) return "";
  return t;
}

export function normalizeSentenceForDedupe(s: string) {
  return stripForSpeech(s)
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/\s+([.!?,])/g, "$1")
    .trim();
}

export function speakText(rawText: string, opts: { ttsMuted: boolean; voice?: SpeechSynthesisVoice | null; interrupt?: boolean }) {
  if (typeof window === "undefined") return;
  if (opts.ttsMuted) return;
  if (!("speechSynthesis" in window)) return;

  const text = stripForSpeech(rawText);
  if (!text) return;

  try {
    if (opts.interrupt) window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.98;
    utterance.pitch = 0.85;
    utterance.volume = 1.0;
    if (opts.voice) utterance.voice = opts.voice;

    window.speechSynthesis.speak(utterance);
  } catch {
    // ignore speech errors
  }
}
