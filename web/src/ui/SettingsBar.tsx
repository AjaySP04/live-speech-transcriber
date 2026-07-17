import { LANGUAGE_NAMES, type Settings } from '../core/types';

/** Model + target-language selectors, plus the processed-languages notice. */
export function SettingsBar({
  settings,
  onChange,
}: {
  settings: Settings | null;
  onChange: (patch: { model?: string; target_lang?: string }) => void;
}) {
  if (!settings) return null;
  return (
    <div className="settings-bar">
      <label>
        Model
        <select
          value={settings.model}
          disabled={settings.loading}
          onChange={(e) => onChange({ model: e.target.value })}
        >
          {settings.available_models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>
      <label>
        My language
        <select
          value={settings.target_lang}
          onChange={(e) => onChange({ target_lang: e.target.value })}
        >
          {settings.allowed_languages.map((l) => (
            <option key={l} value={l}>
              {LANGUAGE_NAMES[l] ?? l}
            </option>
          ))}
        </select>
      </label>
      {settings.loading && <span className="settings-note loading">loading model…</span>}
      {settings.error && <span className="settings-note error">{settings.error}</span>}
      <span className="settings-note">
        Listens for {settings.allowed_languages.map((l) => LANGUAGE_NAMES[l] ?? l).join(', ')} — other
        languages are ignored.
      </span>
      <span className="settings-note legend" title="The dot before each translation shows how sure the transcription model was (its mean token probability).">
        Confidence:
        <span className="conf-dot conf-high" /> high ≥80%
        <span className="conf-dot conf-medium" /> medium 55–80%
        <span className="conf-dot conf-low" /> low &lt;55%
      </span>
    </div>
  );
}
