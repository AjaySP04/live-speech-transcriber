import { describe, expect, it } from 'vitest';
import { confidenceLevel } from './types';

describe('confidenceLevel', () => {
  it('maps high probabilities to high', () => {
    expect(confidenceLevel(0.95)).toBe('high');
    expect(confidenceLevel(0.8)).toBe('high');
  });

  it('maps middling probabilities to medium', () => {
    expect(confidenceLevel(0.79)).toBe('medium');
    expect(confidenceLevel(0.55)).toBe('medium');
  });

  it('maps low probabilities to low', () => {
    expect(confidenceLevel(0.54)).toBe('low');
    expect(confidenceLevel(0)).toBe('low');
  });

  it('treats missing confidence (old rows) as high so history stays calm', () => {
    expect(confidenceLevel(undefined)).toBe('high');
  });
});
