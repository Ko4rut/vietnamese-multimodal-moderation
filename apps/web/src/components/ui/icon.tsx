import type { ReactNode, SVGProps } from "react";

export type IconName =
  | "sparkles" | "layers" | "grid" | "info" | "sun" | "moon" | "arrow-right"
  | "check" | "copy" | "refresh" | "text" | "shield" | "check-circle"
  | "alert-circle" | "chart" | "chevron-right" | "github"
  | "activity" | "zap" | "brain" | "menu" | "close" | "target" | "external";

const shapes: Record<IconName, ReactNode> = {
  sparkles: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"/><path d="m19 17 .7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z"/></>,
  layers: <><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/></>,
  grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8h.01"/></>,
  sun: <><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></>,
  moon: <path d="M20.5 14.5A9 9 0 0 1 9.5 3.5 9 9 0 1 0 20.5 14.5Z"/>,
  "arrow-right": <path d="M5 12h14m-6-6 6 6-6 6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
  refresh: <><path d="M20 11a8 8 0 0 0-14-5L4 8"/><path d="M4 3v5h5M4 13a8 8 0 0 0 14 5l2-2"/><path d="M20 21v-5h-5"/></>,
  text: <path d="M4 6h16M4 11h16M4 16h10M4 21h7"/>,
  shield: <><path d="M12 22c6-2 8-6 8-11V5l-8-3-8 3v6c0 5 2 9 8 11Z"/><path d="m9 12 2 2 4-4"/></>,
  "check-circle": <><circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/></>,
  "alert-circle": <><circle cx="12" cy="12" r="9"/><path d="M12 7v6m0 4h.01"/></>,
  chart: <path d="M4 20V10m6 10V4m6 16v-7m5 7H3"/>,
  "chevron-right": <path d="m9 5 7 7-7 7"/>,
  github: <><path d="M9 19c-4 1-4-2-6-2m12 4v-3a3 3 0 0 0-.9-2.4c3-.4 6.1-1.5 6.1-6.6A5 5 0 0 0 19 5.5 4.6 4.6 0 0 0 18.9 2S17.8 1.6 15 3.6a12 12 0 0 0-6 0C6.2 1.6 5.1 2 5.1 2A4.6 4.6 0 0 0 5 5.5 5 5 0 0 0 3.8 9c0 5 3.1 6.2 6.1 6.6A3 3 0 0 0 9 18v3"/></>,
  activity: <path d="M3 12h4l3-7 4 14 3-7h4"/>,
  zap: <path d="m13 2-9 12h7l-1 8 10-12h-7l0-8Z"/>,
  brain: <><path d="M12 6a3 3 0 0 0-5-2 4 4 0 0 0-3 6 4 4 0 0 0 1 7 4 4 0 0 0 7 3V6Zm0 0a3 3 0 0 1 5-2 4 4 0 0 1 3 6 4 4 0 0 1-1 7 4 4 0 0 1-7 3V6Z"/><path d="M8 9a4 4 0 0 1 4 4m4-4a4 4 0 0 0-4 4"/></>,
  menu: <path d="M4 6h16M4 12h16M4 18h16"/>,
  close: <path d="M5 5l14 14M19 5 5 19"/>,
  target: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
  external: <><path d="M13 4h7v7m0-7-9 9"/><path d="M20 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h5"/></>,
};

interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  size?: number;
}

export function Icon({ name, size = 20, ...props }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"
      aria-hidden="true" {...props}>
      {shapes[name]}
    </svg>
  );
}
