import { describe, expect, it } from 'vitest';
import { speakerNumber, type Utterance } from './types';

const base: Omit<Utterance, 'speaker' | 'speaker_num'> = {
  lang: 'hi',
  original_text: 'नमस्ते',
  english_text: 'Hello',
  start_ms: 0,
  duration_ms: 500,
};

describe('speakerNumber', () => {
  it('uses the live event field', () => {
    expect(speakerNumber({ ...base, speaker: 2 })).toBe(2);
  });

  it('uses the stored-row field', () => {
    expect(speakerNumber({ ...base, speaker_num: 3 })).toBe(3);
  });

  it('prefers the live field when both are present', () => {
    expect(speakerNumber({ ...base, speaker: 2, speaker_num: 9 })).toBe(2);
  });

  it('defaults to 1 when neither is present', () => {
    expect(speakerNumber({ ...base })).toBe(1);
  });
});
