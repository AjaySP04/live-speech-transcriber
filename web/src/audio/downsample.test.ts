import { describe, expect, it } from 'vitest';
import { downsample } from './capture';

describe('downsample', () => {
  it('reduces 48 kHz input to one third the length', () => {
    const input = new Float32Array(4800);
    expect(downsample(input, 48000).length).toBe(1600);
  });

  it('preserves a constant signal', () => {
    const input = new Float32Array(480).fill(0.5);
    const out = downsample(input, 48000);
    for (const s of out) {
      expect(s).toBe(Math.round(0.5 * 32767));
    }
  });

  it('clamps out-of-range samples instead of overflowing', () => {
    const loud = new Float32Array(48).fill(1.5);
    const quiet = new Float32Array(48).fill(-1.5);
    expect(Math.max(...downsample(loud, 48000))).toBe(32767);
    expect(Math.min(...downsample(quiet, 48000))).toBe(-32768);
  });

  it('is the identity mapping at the target rate', () => {
    const input = Float32Array.from([0, 0.25, -0.25, 0.5]);
    const out = downsample(input, 16000);
    expect(Array.from(out)).toEqual([0, 8192, -8192, 16384]);
  });
});
