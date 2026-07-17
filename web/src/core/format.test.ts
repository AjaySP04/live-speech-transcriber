import { describe, expect, it } from 'vitest';
import { formatTime, speakerHue } from './format';

describe('formatTime', () => {
  it('formats zero and sub-second offsets', () => {
    expect(formatTime(0)).toBe('0:00');
    expect(formatTime(999)).toBe('0:00');
  });

  it('formats minute boundaries with zero-padded seconds', () => {
    expect(formatTime(61_000)).toBe('1:01');
    expect(formatTime(60_000)).toBe('1:00');
    expect(formatTime(9_000)).toBe('0:09');
  });

  it('handles long sessions', () => {
    expect(formatTime(3_599_999)).toBe('59:59');
    expect(formatTime(3_600_000)).toBe('60:00');
  });
});

describe('speakerHue', () => {
  it('is stable for a given speaker', () => {
    expect(speakerHue(1)).toBe(speakerHue(1));
  });

  it('gives distinct hues to the first several speakers', () => {
    const hues = [1, 2, 3, 4, 5, 6].map(speakerHue);
    expect(new Set(hues).size).toBe(hues.length);
  });

  it('stays on the color wheel', () => {
    for (let n = 1; n <= 50; n++) {
      expect(speakerHue(n)).toBeGreaterThanOrEqual(0);
      expect(speakerHue(n)).toBeLessThan(360);
    }
  });
});
