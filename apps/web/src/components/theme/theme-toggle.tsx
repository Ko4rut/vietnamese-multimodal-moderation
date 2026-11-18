"use client";

import { Icon } from "@/components/ui/icon";

export function ThemeToggle() {
  function toggleTheme() {
    const root = document.documentElement;
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("viettopic-theme", next);
    } catch {
      // Private browsing may block storage; the current session still works.
    }
  }

  return (
    <button className="icon-button theme-toggle" type="button" onClick={toggleTheme}
      aria-label="Chuyển giữa giao diện sáng và tối" title="Đổi giao diện sáng / tối">
      <Icon name="moon" className="theme-show-dark" size={19} />
      <Icon name="sun" className="theme-show-light" size={19} />
    </button>
  );
}
