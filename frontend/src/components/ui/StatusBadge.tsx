"use client";

export type Status = "created" | "preprocessing" | "training" | "paused" | "done" | "error";

const statusConfig = {
  created: {
    bg: "bg-accent-violet",
    text: "text-background",
    label: "Created",
  },
  preprocessing: {
    bg: "bg-accent-amber",
    text: "text-background",
    label: "Preprocessing",
  },
  training: {
    bg: "bg-accent-emerald",
    text: "text-background",
    label: "Training",
  },
  paused: {
    bg: "bg-card-muted",
    text: "text-foreground",
    label: "Paused",
  },
  done: {
    bg: "bg-accent-emerald",
    text: "text-background",
    label: "Done",
  },
  error: {
    bg: "bg-accent-rose",
    text: "text-background",
    label: "Error",
  },
};

export function StatusBadge({ status }: { status: Status }) {
  const config = statusConfig[status];

  return (
    <span
      className={`inline-block rounded-lg px-3 py-1 text-xs font-medium ${config.bg} ${config.text}`}
    >
      {config.label}
    </span>
  );
}
