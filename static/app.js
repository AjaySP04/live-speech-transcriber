const $ = (id) => document.getElementById(id);
const TARGET_RATE = 16000;
const SEND_BATCH = 2048; // samples (~128 ms)

let ws = null;
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let sessionId = null;
let running = false;
let sendBuf = new Int16Array(0);

// ---------- audio ----------
function downsample(f32, fromRate) {
  const ratio = fromRate / TARGET_RATE;
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

function queueSamples(int16) {
  const merged = new Int16Array(sendBuf.length + int16.length);
  merged.set(sendBuf); merged.set(int16, sendBuf.length);
  sendBuf = merged;
  while (sendBuf.length >= SEND_BATCH) {
    const out = sendBuf.slice(0, SEND_BATCH);
    sendBuf = sendBuf.slice(SEND_BATCH);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(out.buffer);
  }
}

async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    banner(`Microphone access denied or unavailable (${e.name}). ` +
      `Allow mic access in your browser settings. On non-localhost addresses the page must be HTTPS.`);
    return false;
  }
  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule('worklet.js');
  const src = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'capture');
  workletNode.port.onmessage = (e) => queueSamples(downsample(e.data, audioCtx.sampleRate));
  src.connect(workletNode);
  return true;
}

function stopMic() {
  if (workletNode) workletNode.disconnect();
  if (audioCtx) audioCtx.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  workletNode = audioCtx = mediaStream = null;
}

// ---------- websocket ----------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const q = sessionId ? `?session=${sessionId}` : '';
  ws = new WebSocket(`${proto}://${location.host}/ws${q}`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => handleEvent(JSON.parse(e.data));
  ws.onclose = () => {
    if (running) {
      setStatus('reconnecting', 'reconnecting…');
      setTimeout(connect, 1500); // resume same session
    }
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case 'session_started':
      sessionId = ev.session_id;
      setStatus('listening', 'listening');
      break;
    case 'speech_start':
      setStatus('speech', 'hearing speech…');
      break;
    case 'transcribing':
      setStatus('transcribing', 'transcribing…');
      break;
    case 'utterance':
      appendUtterance($('transcript'), ev);
      setStatus('listening', 'listening');
      break;
  }
}

// ---------- rendering ----------
function speakerColor(n) {
  return `hsl(${(n * 67) % 360} 60% 45%)`;
}

function appendUtterance(container, u) {
  const div = document.createElement('div');
  div.className = 'utterance';
  const showOrig = $('show-original').checked && u.original_text;
  const spk = escapeHtml(String(u.speaker ?? u.speaker_num));
  const lang = escapeHtml(u.lang ?? '');
  div.innerHTML = `
    <span class="speaker" style="color:${speakerColor(u.speaker ?? u.speaker_num)}">
      Person ${spk}</span>
    <span class="lang">${lang}</span>
    ${showOrig ? `<div class="original">${escapeHtml(u.original_text)}</div>` : ''}
    <div class="english">${u.original_text ? '↳ ' : ''}${escapeHtml(u.english_text)}</div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s ?? '';
  return d.innerHTML;
}

function setStatus(cls, text) {
  const el = $('status');
  el.className = `status ${cls}`;
  el.textContent = text;
}

function banner(msg) {
  const b = $('banner');
  b.textContent = msg;
  b.classList.remove('hidden');
}

// ---------- controls ----------
$('btn-mic').onclick = async () => {
  if (!running) {
    if (!(await startMic())) return;
    running = true;
    sessionId = null; // new session on manual start
    connect();
    $('btn-mic').textContent = '⏹ Stop';
  } else {
    running = false;
    stopMic();
    if (ws) ws.close();
    setStatus('idle', 'idle');
    $('btn-mic').textContent = '🎤 Start listening';
  }
};

// ---------- history ----------
async function loadSessions() {
  const res = await fetch('/api/sessions');
  const sessions = await res.json();
  const list = $('session-list');
  list.innerHTML = sessions.length ? '' : '<p>No conversations yet.</p>';
  for (const s of sessions) {
    const item = document.createElement('button');
    item.className = 'session-item';
    item.textContent = `#${s.id} ${s.started_at} — ${s.title || '(untitled)'} (${s.utterance_count})`;
    item.onclick = async () => {
      const d = await (await fetch(`/api/sessions/${s.id}`)).json();
      const box = $('session-detail');
      box.innerHTML = `<h3>${escapeHtml(d.title || 'Untitled')}</h3>`;
      for (const u of d.utterances) appendUtterance(box, u);
    };
    list.appendChild(item);
  }
}

$('tab-live').onclick = () => switchTab(true);
$('tab-history').onclick = () => { switchTab(false); loadSessions(); };
function switchTab(live) {
  $('view-live').classList.toggle('hidden', !live);
  $('view-history').classList.toggle('hidden', live);
  $('tab-live').classList.toggle('active', live);
  $('tab-history').classList.toggle('active', !live);
}
