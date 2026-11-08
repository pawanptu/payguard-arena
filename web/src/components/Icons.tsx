import type { ReactNode, SVGProps } from "react";

interface IconProps {
  size?: number;
}

function Svg({ size = 14, children, ...rest }: IconProps & { children: ReactNode } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className="icon"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const CheckIcon = (p: IconProps) => (
  <Svg {...p}><path d="M3 8.5l3.2 3.2L13 4.5" /></Svg>
);
export const CrossIcon = (p: IconProps) => (
  <Svg {...p}><path d="M4 4l8 8M12 4l-8 8" /></Svg>
);
export const WarnIcon = (p: IconProps) => (
  <Svg {...p}><path d="M8 2.2l6.2 11H1.8z" /><path d="M8 6.6v3.2M8 11.8v.1" /></Svg>
);
export const AlertIcon = (p: IconProps) => (
  <Svg {...p}><circle cx="8" cy="8" r="6" /><path d="M8 4.8v3.8M8 11v.1" /></Svg>
);
export const ShieldIcon = (p: IconProps) => (
  <Svg {...p}><path d="M8 1.8l5 1.7v4c0 3-2.1 4.9-5 6-2.9-1.1-5-3-5-6v-4z" /></Svg>
);
export const DiamondIcon = (p: IconProps) => (
  <Svg {...p}><path d="M8 2l5 6-5 6-5-6z" /></Svg>
);
export const SquareIcon = (p: IconProps) => (
  <Svg {...p}><rect x="3.5" y="3.5" width="9" height="9" rx="1.5" /></Svg>
);
export const PlayIcon = (p: IconProps) => (
  <Svg {...p} fill="currentColor"><path d="M5 3l8 5-8 5z" /></Svg>
);
export const PauseIcon = (p: IconProps) => (
  <Svg {...p} fill="currentColor"><rect x="3.5" y="3" width="3" height="10" rx="0.8" /><rect x="9.5" y="3" width="3" height="10" rx="0.8" /></Svg>
);
export const RestartIcon = (p: IconProps) => (
  <Svg {...p}><path d="M2.5 8a5.5 5.5 0 1 0 1.8-4.1" /><path d="M2.5 2.5v3h3" /></Svg>
);
export const RefreshIcon = (p: IconProps) => (
  <Svg {...p}><path d="M13.5 8a5.5 5.5 0 1 1-1.8-4.1" /><path d="M13.5 2.5v3h-3" /></Svg>
);
export const SunIcon = (p: IconProps) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="2.8" />
    <path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" />
  </Svg>
);
export const MoonIcon = (p: IconProps) => (
  <Svg {...p}><path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8z" /></Svg>
);
export const MonitorIcon = (p: IconProps) => (
  <Svg {...p}><rect x="1.8" y="2.5" width="12.4" height="8.4" rx="1.5" /><path d="M5.5 13.5h5M8 10.9v2.6" /></Svg>
);
export const StarIcon = (p: IconProps) => (
  <Svg {...p} fill="currentColor"><path d="M8 1.8l1.9 4 4.3.5-3.2 2.9.9 4.3L8 11.4 4.1 13.5l.9-4.3L1.8 6.3l4.3-.5z" /></Svg>
);
export const CopyIcon = (p: IconProps) => (
  <Svg {...p}><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" /><path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2" /></Svg>
);
export const InfoIcon = (p: IconProps) => (
  <Svg {...p}><circle cx="8" cy="8" r="6" /><path d="M8 7.2v4M8 4.9v.1" /></Svg>
);
export const InboxIcon = (p: IconProps) => (
  <Svg {...p}><path d="M2 9.5l1.7-6a1 1 0 0 1 1-.7h6.6a1 1 0 0 1 1 .7l1.7 6v3a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z" /><path d="M2 9.5h3.5l.7 1.5h3.6l.7-1.5H14" /></Svg>
);
export const ShieldMark = ({ size = 20 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" focusable="false" className="icon">
    <path d="M8 1l6 2v5c0 3.5-2.6 5.7-6 7-3.4-1.3-6-3.5-6-7V3z" fill="currentColor" />
  </svg>
);
