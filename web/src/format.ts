export const pct = (n: number) => `${n.toFixed(1)}%`;
export const num1 = (n: number) => n.toFixed(1);
export const num2 = (n: number) => n.toFixed(2);
export const usd = (n: number) =>
  `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const day = (h: number) => `day ${(h / 24).toFixed(1)}`;
export const splitLabel = (s: string) => s.replace(/_/g, " ");
