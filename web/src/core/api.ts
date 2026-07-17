// History REST client. Platform-neutral (fetch exists in React Native).

import type { SessionDetail, SessionSummary } from './types';

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
