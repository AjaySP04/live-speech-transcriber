// History REST client. Platform-neutral (fetch exists in React Native).

import type { SessionDetail, SessionSummary, Settings } from './types';

export async function fetchSessions(baseUrl = ''): Promise<SessionSummary[]> {
  const res = await fetch(`${baseUrl}/api/sessions`);
  if (!res.ok) throw new Error(`sessions request failed: ${res.status}`);
  return res.json();
}

export async function fetchSession(id: number, baseUrl = ''): Promise<SessionDetail> {
  const res = await fetch(`${baseUrl}/api/sessions/${id}`);
  if (!res.ok) throw new Error(`session ${id} request failed: ${res.status}`);
  return res.json();
}

export async function fetchSettings(baseUrl = ''): Promise<Settings> {
  const res = await fetch(`${baseUrl}/api/settings`);
  if (!res.ok) throw new Error(`settings request failed: ${res.status}`);
  return res.json();
}

export async function updateSettings(
  patch: { model?: string; target_lang?: string },
  baseUrl = '',
): Promise<Settings> {
  const res = await fetch(`${baseUrl}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
