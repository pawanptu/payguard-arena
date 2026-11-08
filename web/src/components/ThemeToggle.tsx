import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { MonitorIcon, MoonIcon, SunIcon } from "./Icons";

type Theme = "system" | "light" | "dark";
const KEY = "payguard-theme";
const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const NAME: Record<Theme, string> = { system: "System", light: "Light", dark: "Dark" };
const ICON: Record<Theme, ReactElement> = {
  system: <MonitorIcon size={18} />,
  light: <SunIcon size={18} />,
  dark: <MoonIcon size={18} />,
};

function read(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* storage may be unavailable (private mode, blocked cookies) */
  }
  return "system";
}

/** 3-state theme button: system (default), light, dark. Only sets data-theme; CSS light-dark() does the rest. */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(read);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const text = `Theme: ${NAME[theme]}. Activate to switch to ${NAME[NEXT[theme]]}.`;
  return (
    <button type="button" className="icon-btn" aria-label={text} title={`Theme: ${NAME[theme]}`} onClick={() => setTheme(NEXT[theme])}>
      {ICON[theme]}
    </button>
  );
}
