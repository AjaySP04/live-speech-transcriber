import type { CSSProperties } from 'react';
import { confidenceLevel, speakerNumber, LANGUAGE_NAMES, type Utterance } from '../core/types';
import { formatTime, speakerHue } from '../core/format';

/** One utterance: the translation is the primary line, the original script
 *  beneath it. The dot before the translation encodes whisper's confidence. */
export function Couplet({ u, targetLang }: { u: Utterance; targetLang: string }) {
  const n = speakerNumber(u);
  const style = { '--spk-h': speakerHue(n) } as CSSProperties;
  const conf = confidenceLevel(u.confidence);
  const confPct = u.confidence !== undefined ? `${Math.round(u.confidence * 100)}%` : '–';
  const translatedTo = u.translated_to ?? 'en';
  const isFallback = !u.blocked && translatedTo !== targetLang;
  const latency = u.latency_ms !== undefined ? ` · translated in ${(u.latency_ms / 1000).toFixed(1)}s` : '';

  if (u.blocked) {
    return (
      <article className="couplet no-original blocked" style={style}>
        <div className="meta">
          <span className="speaker">Person {n}</span>
          <span className="lang">{u.lang}</span>
          <span className="fallback">content withheld</span>
          <span className="time">{formatTime(u.start_ms)}</span>
        </div>
        <p className="english blocked-line">{u.english_text}</p>
      </article>
    );
  }

  return (
    <article className={`couplet${u.original_text ? '' : ' no-original'}`} style={style}>
      <div className="meta">
        <span className="speaker">Person {n}</span>
        <span className="lang">{u.lang} → {translatedTo}</span>
        {isFallback && (
          <span className="fallback" title={`${LANGUAGE_NAMES[targetLang] ?? targetLang} translation is not available yet; shown in English`}>
            English fallback
          </span>
        )}
        <span className="time" title={`confidence ${confPct}${latency}`}>
          {formatTime(u.start_ms)}
        </span>
      </div>
      <p className={`english conf-${conf}`} title={`Translated to ${LANGUAGE_NAMES[translatedTo] ?? translatedTo} · confidence ${confPct}${latency}`}>
        <span className={`conf-dot conf-${conf}`} aria-label={`confidence ${confPct}`} />
        {u.english_text}
      </p>
      {u.original_text && (
        <p className="original" dir="auto">
          {u.original_text}
        </p>
      )}
    </article>
  );
}
