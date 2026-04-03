import { ReactNode } from "react";

export function StatCard({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="card-base flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-card-muted">
          {icon}
        </div>
        <p className="text-sm text-foreground/60">{label}</p>
      </div>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}
