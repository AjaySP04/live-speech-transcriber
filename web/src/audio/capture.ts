// Web microphone capture: getUserMedia + AudioWorklet, downsampled to
// 16 kHz mono i16. This module is DOM-specific — the React Native app will
// have its own capture implementation feeding the same core/ws.ts client.

export interface Capture {
  stop(): void;
}

export class MicPermissionError extends Error {}

export function downsample(f32: Float32Array, fromRate: number, toRate = 16000): Int16Array {
  const ratio = fromRate / toRate;
  const outLen = Math.floor(f32.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, f32.length - 1);
    const s = f32[i0] + (f32[i1] - f32[i0]) * (pos - i0);
    out[i] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
  }
  return out;
}

export async function startCapture(onPcm: (samples: Int16Array) => void): Promise<Capture> {
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    throw new MicPermissionError(e instanceof Error ? e.name : String(e));
  }

  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule('/worklet.js');
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'capture');
  node.port.onmessage = (e) => onPcm(downsample(e.data as Float32Array, ctx.sampleRate));
  src.connect(node);

  return {
    stop() {
      node.disconnect();
      void ctx.close();
      stream.getTracks().forEach((t) => t.stop());
    },
  };
}
