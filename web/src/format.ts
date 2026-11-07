export const pct = (n: number) => `${n.toFixed(1)}%`;
export const num1 = (n: number) => n.toFixed(1);
export const num2 = (n: number) => n.toFixed(2);
export const usd = (n: number) =>
  `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const day = (h: number) => `day ${(h / 24).toFixed(1)}`;
export const splitLabel = (s: string) => s.replace(/_/g, " ");
/** "PB-1234567890ABCDEF" -> "PB-1234...CDEF"; short ids are left alone. */
export const shortId = (id: string) => (id.length > 14 ? `${id.slice(0, 7)}...${id.slice(-4)}` : id);
/** "2026-11-07T12:00:01.123456Z" -> "2026-11-07 12:00:01 UTC"; anything else is shown as given. */
export const shortTime = (iso: string) => {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})/.exec(iso);
  return m && /(Z|[+-]00:?00)$/.test(iso) ? `${m[1]} ${m[2]} UTC` : iso;
};
