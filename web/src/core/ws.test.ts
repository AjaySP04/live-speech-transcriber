import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TranscriberSocket } from './ws';
import type { PipelineEvent } from './types';

/** Controllable stand-in for the global WebSocket. */
class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.OPEN;
  binaryType = '';
  sent: ArrayBuffer[] = [];
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: ArrayBuffer) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }

  receive(ev: PipelineEvent) {
    this.onmessage?.({ data: JSON.stringify(ev) });
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal('WebSocket', FakeWebSocket);
  vi.useFakeTimers();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

function newSocket(events: PipelineEvent[] = []) {
  const socket = new TranscriberSocket({
    baseUrl: 'ws://test',
    onEvent: (ev) => events.push(ev),
  });
  socket.start();
  return { socket, ws: () => FakeWebSocket.instances.at(-1)! };
}

describe('TranscriberSocket audio batching', () => {
  it('holds samples until a full 2048-sample batch is available', () => {
    const { socket, ws } = newSocket();
    socket.sendAudio(new Int16Array(2000));
    expect(ws().sent).toHaveLength(0);
    socket.sendAudio(new Int16Array(48));
    expect(ws().sent).toHaveLength(1);
    expect(ws().sent[0].byteLength).toBe(2048 * 2);
  });

  it('drains multiple batches from one large buffer and keeps the remainder', () => {
    const { socket, ws } = newSocket();
    socket.sendAudio(new Int16Array(5000)); // 2 batches, 904 left over
    expect(ws().sent).toHaveLength(2);
    socket.sendAudio(new Int16Array(1144)); // 904 + 1144 = one more batch exactly
    expect(ws().sent).toHaveLength(3);
  });
});

describe('TranscriberSocket reconnect', () => {
  it('reconnects into the same session after a drop', () => {
    const events: PipelineEvent[] = [];
    const { ws } = newSocket(events);
    expect(ws().url).toBe('ws://test/ws');

    ws().receive({ type: 'session_started', session_id: 42 });
    ws().close(); // unexpected drop

    vi.advanceTimersByTime(1500);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(ws().url).toBe('ws://test/ws?session=42');
  });

  it('does not reconnect after an intentional stop', () => {
    const { socket } = newSocket();
    socket.stop();
    vi.advanceTimersByTime(5000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it('forwards events to the handler', () => {
    const events: PipelineEvent[] = [];
    const { ws } = newSocket(events);
    ws().receive({ type: 'speech_start' });
    expect(events).toEqual([{ type: 'speech_start' }]);
  });
});
