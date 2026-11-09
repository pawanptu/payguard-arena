import { Skeleton } from "./ui";

const Hidden = ({ what }: { what: string }) => <span className="visually-hidden">Loading {what}</span>;

export function FeedSkeleton() {
  return (
    <div className="card" aria-busy="true">
      <Hidden what="replay feed" />
      <Skeleton className="skel-title" />
      <Skeleton className="skel-controls" />
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <Skeleton key={i} className="skel-row" />
      ))}
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="card" aria-busy="true">
      <Hidden what="evolution chart" />
      <Skeleton className="skel-title" />
      <Skeleton className="skel-chart" />
    </div>
  );
}

export function TableSkeleton({ what, rows = 8 }: { what: string; rows?: number }) {
  return (
    <div className="card" aria-busy="true">
      <Hidden what={what} />
      <Skeleton className="skel-title" />
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="skel-row skel-row-table" />
      ))}
    </div>
  );
}
