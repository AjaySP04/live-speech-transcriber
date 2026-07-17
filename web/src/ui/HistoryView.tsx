import { useEffect, useState } from 'react';
import { fetchSession, fetchSessions } from '../core/api';
import type { SessionDetail, SessionSummary } from '../core/types';
import { Couplet } from './Couplet';

export function HistoryView({ showOriginal }: { showOriginal: boolean }) {
  const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSessions()
      .then(setSessions)
      .catch(() => setError('Could not load conversations. Is the server running?'));
  }, []);

  const open = (id: number) => {
    setError(null);
    fetchSession(id)
      .then(setDetail)
      .catch(() => setError('Could not load that conversation.'));
  };

  return (
    <main>
      {error && <div className="banner">{error}</div>}

      <div className="session-list">
        {sessions?.length === 0 && (
          <p className="empty">No conversations yet. Everything you transcribe is saved here.</p>
        )}
        {sessions?.map((s) => (
          <button key={s.id} className="session-item" onClick={() => open(s.id)}>
            <span className="s-title">{s.title || 'Untitled conversation'}</span>
            <span className="s-meta">
              {s.started_at} · {s.utterance_count} utterance{s.utterance_count === 1 ? '' : 's'}
            </span>
          </button>
        ))}
      </div>

      {detail && (
        <div className={`transcript${showOriginal ? '' : ' hide-original'}`}>
          <h3 className="detail-title">{detail.title || 'Untitled conversation'}</h3>
          {detail.utterances.map((u, i) => (
            <Couplet key={i} u={u} />
          ))}
        </div>
      )}
    </main>
  );
}
