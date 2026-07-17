// Theme override: follows the system by default; a manual choice is stamped
// on <html data-theme> and persisted. Cycle: system → light → dark → system.

import { useCallback, useState } from 'react';

const THEME_KEY = 'tarjuman-theme';
export type Theme = 'light' | 'dark' | null;

function apply(t: Theme) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}

export function useTheme(): { theme: Theme; cycle: () => void } {
  const [theme, setTheme] = useState<Theme>(
    () => localStorage.getItem(THEME_KEY) as Theme,
  );

  const cycle = useCallback(() => {
    setTheme((cur) => {
      const next: Theme = cur === null ? 'light' : cur === 'light' ? 'dark' : null;
      if (next) localStorage.setItem(THEME_KEY, next);
      else localStorage.removeItem(THEME_KEY);
      apply(next);
      return next;
    });
  }, []);

  return { theme, cycle };
}
