import { useEffect, useRef } from 'react';
import { Couplet } from './Couplet';
import type { TranscriberState } from './useTranscriber';
import type { Status } from '../core/types';

const STATUS_LABEL: Record<Status, string> = {
  idle: 'Idle',
  listening: 'Listening',
  speech: 'Hearing speech',
  transcribing: 'Transcribing…',
  reconnecting: 'Reconnecting…',
};

export function LiveView({
  t,
  showOriginal,
  onShowOriginal,
}: {
  t: TranscriberState;
  showOriginal: boolean;
  onShowOriginal: (v: boolean) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'nearest' });
  }, [t.utterances.length]);

  return (
    <main>
      <div className="controls">
        <button className={`mic${t.running ? ' running' : ''}`} onClick={() => void t.toggle()}>
          {t.running ? 'Stop' : 'Start listening'}
        </button>
        <span className={`status ${t.status}`}>
          <span className="dot" />
          <span>{STATUS_LABEL[t.status]}</span>
        </span>
        <label className="switch">
          <input
            type="checkbox"
            checked={showOriginal}
            onChange={(e) => onShowOriginal(e.target.checked)}
          />
          <span>Original text</span>
        </label>
      </div>

      {t.error && <div className="banner">{t.error}</div>}

      <div className={`transcript${showOriginal ? '' : ' hide-original'}`}>
        {t.utterances.length === 0 ? (
          <p className="empty">
            Press “Start listening” and speak — Hindi, Urdu, Arabic, or any language. Each line
            shows the original script with its English translation.
          </p>
        ) : (
          t.utterances.map((u, i) => <Couplet key={i} u={u} />)
        )}
        <div ref={endRef} />
      </div>
    </main>
  );
}
