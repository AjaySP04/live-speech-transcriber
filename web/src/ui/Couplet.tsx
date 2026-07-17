import type { CSSProperties } from 'react';
import { speakerNumber, type Utterance } from '../core/types';
import { formatTime, speakerHue } from '../core/format';

/** One utterance: the original script above, its English interpretation beneath. */
export function Couplet({ u }: { u: Utterance }) {
  const n = speakerNumber(u);
  const style = { '--spk-h': speakerHue(n) } as CSSProperties;
  return (
    <article className={`couplet${u.original_text ? '' : ' no-original'}`} style={style}>
      <div className="meta">
        <span className="speaker">Person {n}</span>
        <span className="lang">{u.lang}</span>
        <span className="time">{formatTime(u.start_ms)}</span>
      </div>
      {u.original_text && (
        <p className="original" dir="auto">
          {u.original_text}
        </p>
      )}
      <p className="english">{u.english_text}</p>
    </article>
  );
}
