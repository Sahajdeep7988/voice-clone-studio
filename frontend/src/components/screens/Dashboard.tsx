"use client";

import Link from "next/link";
import { Plus, TrendingDown, Zap, AlertCircle } from "lucide-react";
import { mockSessions, mockModels, mockSystemStats } from "@/lib/mockData";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { StatCard } from "@/components/ui/StatCard";

export function Dashboard() {
  const activeSessions = mockSessions.filter(
    (s) => s.status === "training" || s.status === "preprocessing"
  );
  const completedModels = mockModels.filter((m) => m.status === "done");

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="mt-2 text-foreground/60">
            Monitor your training sessions and manage voice models
          </p>
        </div>
        <Link
          href="/session-builder"
          className="flex items-center gap-2 rounded-lg bg-accent-amber px-6 py-3 font-medium text-background transition-opacity hover:opacity-90"
        >
          <Plus size={20} />
          New Training
        </Link>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        <StatCard
          icon={<Zap className="text-accent-amber" />}
          label="GPU VRAM"
          value={mockSystemStats.gpuVram}
        />
        <StatCard
          icon={<AlertCircle className="text-accent-emerald" />}
          label="GPU Status"
          value={mockSystemStats.gpuStatus}
        />
        <StatCard
          icon={<TrendingDown className="text-accent-violet" />}
          label="Total Models"
          value={mockSystemStats.totalModels.toString()}
        />
        <StatCard
          icon={<TrendingDown className="text-accent-rose" />}
          label="Active Sessions"
          value={activeSessions.length.toString()}
        />
      </div>

      {/* Active Training Sessions */}
      <section>
        <h2 className="mb-4 text-xl font-semibold">Active Training Sessions</h2>
        {activeSessions.length > 0 ? (
          <div className="grid gap-4">
            {activeSessions.map((session) => (
              <Link
                key={session.id}
                href={`/training/${session.id}`}
                className="card-base flex items-center justify-between hover:border-accent-amber transition-all"
              >
                <div className="flex-1">
                  <h3 className="font-semibold">{session.modelName}</h3>
                  <p className="mt-1 text-sm text-foreground/60">
                    Epoch {session.epochs.current} / {session.epochs.total} • Loss:{" "}
                    {session.loss.toFixed(4)}
                  </p>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-card-muted">
                    <div
                      className="h-full bg-accent-emerald transition-all"
                      style={{
                        width: `${
                          (session.epochs.current / session.epochs.total) * 100
                        }%`,
                      }}
                    />
                  </div>
                </div>
                <StatusBadge status={session.status} />
              </Link>
            ))}
          </div>
        ) : (
          <div className="card-base text-center py-12">
            <p className="text-foreground/60">No active training sessions</p>
            <Link
              href="/session-builder"
              className="mt-4 inline-block text-accent-amber hover:underline"
            >
              Start a new training
            </Link>
          </div>
        )}
      </section>

      {/* Recent Models */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent Models</h2>
          <Link href="/my-models" className="text-sm text-accent-amber hover:underline">
            View all
          </Link>
        </div>
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
          {mockModels.slice(0, 4).map((model) => (
            <div
              key={model.id}
              className="card-base flex flex-col"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="font-semibold">{model.name}</h3>
                  <p className="mt-1 text-xs text-foreground/60">
                    {model.epochs.current} / {model.epochs.total} epochs
                  </p>
                </div>
                <StatusBadge status={model.status} />
              </div>
              <p className="mt-3 text-xs text-foreground/60">
                Duration: {model.duration}m • Device: {model.device}
              </p>
              {!model.isLocal && (
                <p className="mt-2 text-xs text-accent-rose">
                  ⚠ Weights not found locally
                </p>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
