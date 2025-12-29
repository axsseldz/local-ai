import { ChatMessage } from "@/types/chat";

export function formatAssistantContent(message: ChatMessage) {
  if (message.role !== "assistant") return message.content;
  const text = message.content || "";
  if (!text) return "";
  return text.replace(/\r\n/g, "\n").replace(/\\n/g, "\n");
}

export function uid(prefix = "m") {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function renderPromptPreview(text: string) {
  const tokens = text.split(/(\s+)/);
  return tokens.map((token, idx) => {
    if (token === "/summary") {
      return (
        <span
          key={`cmd_${idx}`}
          className="rounded-sm bg-emerald-500/10 text-emerald-200/90"
        >
          {token}
        </span>
      );
    }
    return <span key={`txt_${idx}`}>{token}</span>;
  });
}

export function buildSummaryPreview(text: string) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return "Generating summary...";
  return cleaned.length > 140 ? `${cleaned.slice(0, 140)}...` : cleaned;
}

export function buildConversationTranscript(messages: ChatMessage[]) {
  return messages
    .map((m) => {
      const role = m.role === "user" ? "User" : "Assistant";
      return `${role}:\n${m.content}`;
    })
    .join("\n\n");
}

export function buildSummaryPrompt(transcript: string, focus: string) {
  const focusLine = focus ? `Focus: ${focus}\n` : "";
  return [
    "You are preparing a highly detailed conversation summary.",
    "Include concrete decisions, key ideas, and any code snippets in fenced blocks.",
    "Use clear markdown sections and bullet lists where helpful.",
    "Do not mention that you are an AI or that you summarized.",
    "",
    focusLine,
    "Conversation:",
    transcript,
  ]
    .filter(Boolean)
    .join("\n");
}

export function extractSummaryTitle(text: string) {
  const match = text.match(/^Title:\s*(.+)$/m);
  if (!match) return "";
  return match[1].trim();
}

export function stripSummaryTitle(text: string) {
  return text.replace(/^Title:.*\n?/m, "").trimStart();
}

export function stripLeadingMeta(text: string) {
  return text
    .replace(/^\s*\{[\s\S]*?\}\s*/m, "")
    .replace(/^#\s*Conversation Summary.*\n?/i, "")
    .replace(/^\s*Conversation Summary:.*\n?/i, "");
}

export function enforceSingleAgentCommand(input: string) {
  const pattern = /(^|\s)\/summary(?=\s|$)/g;
  const matches = [...input.matchAll(pattern)];
  if (matches.length <= 1) return input;
  let kept = 0;
  return input.replace(pattern, (match) => {
    if (kept === 0) {
      kept += 1;
      return match;
    }
    return match.replace("/summary", "").replace(/\s{2,}/g, " ");
  });
}

export function formatBytes(bytes?: number | null) {
  if (bytes == null || Number.isNaN(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}
