import { useCallback, useEffect, useRef, useState } from 'react';
import { LiveView } from './ui/LiveView';
import { HistoryView } from './ui/HistoryView';
import { SettingsBar } from './ui/SettingsBar';
import { useTranscriber } from './ui/useTranscriber';
import { useTheme } from './ui/theme';
import { fetchSettings, updateSettings } from './core/api';
import type { Settings } from './core/types';

export default function App() {
  const [tab, setTab] = useState<'live' | 'history'>('live');
  const [showOriginal, setShowOriginal] = useState(true);
  const [settings, setSettings] = useState<Settings | null>(null);
  const t = useTranscriber();
  const { theme, cycle } = useTheme();
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    fetchSettings().then(setSettings).catch(() => {});
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const changeSettings = useCallback((patch: { model?: string; target_lang?: string }) => {
    updateSettings(patch)
      .then((s) => {
        setSettings(s);
        // A model switch loads in the background; poll until it lands.
        if (s.loading && !pollRef.current) {
          pollRef.current = window.setInterval(async () => {
            const cur = await fetchSettings().catch(() => null);
            if (cur) setSettings(cur);
            if (cur && !cur.loading && pollRef.current) {
              window.clearInterval(pollRef.current);
              pollRef.current = null;
            }
          }, 2000);
        }
      })
      .catch((e) => alert(`Could not update settings: ${e.message}`));
  }, []);

  const targetLang = settings?.target_lang ?? 'en';

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

      <SettingsBar settings={settings} onChange={changeSettings} />

      {tab === 'live' ? (
        <LiveView
          t={t}
          showOriginal={showOriginal}
          onShowOriginal={setShowOriginal}
          targetLang={targetLang}
        />
      ) : (
        <HistoryView showOriginal={showOriginal} targetLang={targetLang} />
      )}
    </>
  );
}
