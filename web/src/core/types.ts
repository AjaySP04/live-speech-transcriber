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
