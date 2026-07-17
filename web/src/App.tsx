import { useState } from 'react';
import { LiveView } from './ui/LiveView';
import { HistoryView } from './ui/HistoryView';
import { useTranscriber } from './ui/useTranscriber';
import { useTheme } from './ui/theme';

export default function App() {
  const [tab, setTab] = useState<'live' | 'history'>('live');
  const [showOriginal, setShowOriginal] = useState(true);
  const t = useTranscriber();
  const { theme, cycle } = useTheme();

  return (
    <>
      <header>
        <h1 className="wordmark">
          <img src="/logo.svg" alt="" className="logo" />
          <span className="wm-ar" lang="ur" dir="rtl">
            ترجمان
          </span>
          <span className="wm-en">tarjuman</span>
        </h1>
        <nav>
          <button className={`tab${tab === 'live' ? ' active' : ''}`} onClick={() => setTab('live')}>
            Live
          </button>
          <button
            className={`tab${tab === 'history' ? ' active' : ''}`}
            onClick={() => setTab('history')}
          >
            History
          </button>
          <button
            className="theme"
            onClick={cycle}
            title={theme ? `Theme: ${theme}. Click to change.` : 'Theme: follows system. Click to override.'}
          >
            {theme === 'light' ? '☀' : theme === 'dark' ? '☾' : '◐'}
          </button>
        </nav>
      </header>

      {tab === 'live' ? (
        <LiveView t={t} showOriginal={showOriginal} onShowOriginal={setShowOriginal} />
      ) : (
        <HistoryView showOriginal={showOriginal} />
      )}
    </>
  );
}
