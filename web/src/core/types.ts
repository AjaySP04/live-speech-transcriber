// Shared domain types — mirror the Rust server's JSON shapes exactly.
// This module is platform-neutral (no DOM): reusable from React Native.

export interface Utterance {
  speaker?: number;      // live WebSocket events
  speaker_num?: number;  // stored rows from the history API
  lang: string;
  original_text: string;
  english_text: string;
  start_ms: number;
  duration_ms: number;
  /** Mean whisper token probability (0–1) for the translation line. */
  confidence?: number;
  /** Language the translation line is actually in ("en" fallback possible). */
  translated_to?: string;
  /** Milliseconds from utterance-close to translation delivery. */
  latency_ms?: number;
  /** True when a guardrail withheld the content (text is a placeholder). */
  blocked?: boolean;
}

export interface Settings {
  model: string;
  target_lang: string;
  allowed_languages: string[];
  available_models: string[];
  loading: boolean;
  error: string | null;
}

/** Display names for the languages the app processes. */
export const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  hi: 'हिन्दी (Hindi)',
  ur: 'اردو (Urdu)',
  ar: 'العربية (Arabic)',
};

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export function confidenceLevel(c: number | undefined): ConfidenceLevel {
  if (c === undefined || c >= 0.8) return 'high';
  if (c >= 0.55) return 'medium';
  return 'low';
}

export type PipelineEvent =
  | { type: 'session_started'; session_id: number }
  | { type: 'speech_start' }
  | { type: 'transcribing' }
  | ({ type: 'utterance' } & Utterance);

export interface SessionSummary {
  id: number;
  started_at: string;
  ended_at: string | null;
  title: string;
  utterance_count: number;
}

export interface SessionDetail {
  id: number;
  started_at: string | null;
  title: string | null;
  utterances: Utterance[];
}

export type Status =
  | 'idle'
  | 'listening'
  | 'speech'
  | 'transcribing'
  | 'reconnecting';

export function speakerNumber(u: Utterance): number {
  return u.speaker ?? u.speaker_num ?? 1;
}
