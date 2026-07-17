// Live-transcription WebSocket client.
// Platform-neutral (no DOM): WebSocket and setTimeout exist in React Native too.

import type { PipelineEvent } from './types';

const SEND_BATCH = 2048; // samples (~128 ms at 16 kHz)
const RECONNECT_DELAY_MS = 1500;

export interface TranscriberSocketOptions {
  /** e.g. "ws://localhost:8080" — no trailing slash */
  baseUrl: string;
  onEvent: (ev: PipelineEvent) => void;
  onReconnecting?: () => void;
}

/**
 * Owns the socket lifecycle: connects, batches outgoing PCM, parses incoming
 * events, and transparently reconnects into the same session on drops.
 */
export class TranscriberSocket {
  private ws: WebSocket | null = null;
  private sessionId: number | null = null;
  private closed = false;
  private sendBuf = new Int16Array(0);

  constructor(private readonly opts: TranscriberSocketOptions) {}

  start(): void {
    this.closed = false;
    this.sessionId = null;
    this.connect();
  }

  stop(): void {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
    this.sendBuf = new Int16Array(0);
  }

  /** Queue 16 kHz mono i16 samples; sent in SEND_BATCH chunks. */
  sendAudio(samples: Int16Array): void {
    const merged = new Int16Array(this.sendBuf.length + samples.length);
    merged.set(this.sendBuf);
    merged.set(samples, this.sendBuf.length);
    this.sendBuf = merged;
    while (this.sendBuf.length >= SEND_BATCH) {
      const chunk = this.sendBuf.slice(0, SEND_BATCH);
      this.sendBuf = this.sendBuf.slice(SEND_BATCH);
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(chunk.buffer);
      }
    }
  }

  private connect(): void {
    const q = this.sessionId != null ? `?session=${this.sessionId}` : '';
    const ws = new WebSocket(`${this.opts.baseUrl}/ws${q}`);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (e) => {
      const ev = JSON.parse(String(e.data)) as PipelineEvent;
      if (ev.type === 'session_started') this.sessionId = ev.session_id;
      this.opts.onEvent(ev);
    };
    ws.onclose = () => {
      if (this.closed) return;
      this.opts.onReconnecting?.();
      setTimeout(() => {
        if (!this.closed) this.connect(); // resumes the same session
      }, RECONNECT_DELAY_MS);
    };
    this.ws = ws;
  }
}
