"use client";

import Link from "next/link";
import { Plus, Download, AlertCircle } from "lucide-react";
import { mockModels } from "@/lib/mockData";
import { StatusBadge } from "@/components/ui/StatusBadge";

export function MyModels() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">My Models</h1>
          <p className="mt-2 text-foreground/60">
            Manage your trained voice models and synced sessions
          </p>
        </div>
        <Link
          href="/session-builder"
          className="flex items-center gap-2 rounded-lg bg-accent-amber px-6 py-3 font-medium text-background transition-opacity hover:opacity-90"
        >
          <Plus size={20} />
          New Model
        </Link>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 border-b border-border">
        {["All", "Local", "Cloud"].map((tab) => (
          <button
            key={tab}
            className={`px-4 py-3 font-medium text-sm transition-colors ${
              tab === "All"
                ? "border-b-2 border-accent-amber text-foreground"
                : "text-foreground/60 hover:text-foreground"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {mockModels.map((model) => (
          <div key={model.id} className="card-base flex flex-col">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold">{model.name}</h3>
                <p className="text-xs text-foreground/60 mt-1">
                  {model.epochs.current} / {model.epochs.total} epochs
                </p>
              </div>
              <StatusBadge status={model.status} />
            </div>

            {/* Details */}
            <div className="space-y-2 border-t border-border pt-3 mb-4 flex-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-foreground/60">Duration</span>
                <span>{model.duration}m</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-foreground/60">Device</span>
                <span>{model.device}</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-foreground/60">Location</span>
                <span>{model.isLocal ? "Local" : "Cloud"}</span>
              </div>
            </div>

            {/* Status Alert */}
            {!model.isLocal && (
              <div className="flex gap-2 items-start rounded-lg bg-card-muted p-3 mb-4 border-l-4 border-accent-rose">
                <AlertCircle size={16} className="text-accent-rose flex-shrink-0 mt-0.5" />
                <p className="text-xs text-foreground/70">
                  Weights not found locally. Download to use for inference.
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Link
                href={`/inference?model=${model.id}`}
                className="flex-1 rounded-lg border border-border px-3 py-2 text-center text-sm font-medium transition-colors hover:bg-card-muted"
              >
                Convert
              </Link>
              {!model.isLocal && (
                <button className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-accent-violet px-3 py-2 text-center text-sm font-medium text-background transition-opacity hover:opacity-90">
                  <Download size={16} />
                  Download
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {mockModels.length === 0 && (
        <div className="card-base text-center py-16">
          <p className="text-lg font-semibold">No models yet</p>
          <p className="mt-2 text-foreground/60">Start by training your first voice model</p>
          <Link
            href="/session-builder"
            className="mt-6 inline-block rounded-lg bg-accent-amber px-6 py-3 font-medium text-background transition-opacity hover:opacity-90"
          >
            Create New Model
          </Link>
        </div>
      )}
    </div>
  );
}
