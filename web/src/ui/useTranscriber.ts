// Web glue: binds mic capture (audio/) to the transcriber socket (core/)
// and exposes React state for the live view.

import { useCallback, useRef, useState } from 'react';
import { TranscriberSocket } from '../core/ws';
import type { Status, Utterance } from '../core/types';
import { startCapture, MicPermissionError, type Capture } from '../audio/capture';

export interface TranscriberState {
  running: boolean;
  status: Status;
  utterances: Utterance[];
  error: string | null;
  toggle: () => Promise<void>;
}

function wsBaseUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}`;
}

export function useTranscriber(): TranscriberState {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<Status>('idle');
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<TranscriberSocket | null>(null);
  const captureRef = useRef<Capture | null>(null);

  const toggle = useCallback(async () => {
    if (captureRef.current) {
      captureRef.current.stop();
      captureRef.current = null;
      socketRef.current?.stop();
      socketRef.current = null;
      setRunning(false);
      setStatus('idle');
      return;
    }

    setError(null);
    const socket = new TranscriberSocket({
      baseUrl: wsBaseUrl(),
      onEvent: (ev) => {
        switch (ev.type) {
          case 'session_started':
            setStatus('listening');
            break;
          case 'speech_start':
            setStatus('speech');
            break;
          case 'transcribing':
            setStatus('transcribing');
            break;
          case 'utterance':
            setUtterances((u) => [...u, ev]);
            setStatus('listening');
            break;
        }
      },
      onReconnecting: () => setStatus('reconnecting'),
    });

    try {
      captureRef.current = await startCapture((pcm) => socket.sendAudio(pcm));
    } catch (e) {
      if (e instanceof MicPermissionError) {
        setError(
          `Microphone access denied or unavailable (${e.message}). ` +
            `Allow mic access in your browser settings. On non-localhost addresses the page must be HTTPS.`,
        );
        return;
      }
      throw e;
    }
    socketRef.current = socket;
    socket.start();
    setUtterances([]);
    setRunning(true);
  }, []);

  return { running, status, utterances, error, toggle };
}
