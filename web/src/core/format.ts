// Presentation helpers shared by web and (later) React Native.

/** Stable per-speaker hue, degrees on the color wheel. */
export function speakerHue(n: number): number {
  return (n * 67) % 360;
}

/** "m:ss" from a millisecond offset. */
export function formatTime(ms: number): string {
  const s = Math.floor((ms ?? 0) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
