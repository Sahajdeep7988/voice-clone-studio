"use client";

import { useState } from "react";
import { Play, X, Check, AlertCircle } from "lucide-react";
import { mockSegments } from "@/lib/mockData";

interface SegmentState {
  [key: string]: boolean;
}

export function Curation({ sessionId }: { sessionId: string }) {
  const [segments, setSegments] = useState<SegmentState>(
    mockSegments.reduce(
      (acc, seg) => ({ ...acc, [seg.id]: seg.approved }),
      {} as SegmentState
    )
  );

  const approvedCount = Object.values(segments).filter(Boolean).length;
  const totalDuration = (approvedCount * 4.5).toFixed(1); // Average segment duration

  const toggleSegment = (id: string) => {
    setSegments((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleApproveAll = () => {
    setSegments(
      mockSegments.reduce((acc, seg) => ({ ...acc, [seg.id]: true }), {} as SegmentState)
    );
  };

  const handleRejectAll = () => {
    setSegments(
      mockSegments.reduce((acc, seg) => ({ ...acc, [seg.id]: false }), {} as SegmentState)
    );
  };

  const handleProceed = () => {
    console.log("[v0] Approved segments:", Object.keys(segments).filter((k) => segments[k]));
    alert(`Proceeding with ${approvedCount} approved segments`);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Segment Curation</h1>
        <p className="mt-2 text-foreground/60">
          Review extracted audio segments and approve/reject them before training
        </p>
      </div>

      {/* Summary */}
      <div className="card-base space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-xs text-foreground/60 mb-1">Total Segments</p>
            <p className="text-2xl font-bold">{mockSegments.length}</p>
          </div>
          <div>
            <p className="text-xs text-foreground/60 mb-1">Approved</p>
            <p className="text-2xl font-bold text-accent-emerald">{approvedCount}</p>
          </div>
          <div>
            <p className="text-xs text-foreground/60 mb-1">Rejected</p>
            <p className="text-2xl font-bold text-accent-rose">
              {mockSegments.length - approvedCount}
            </p>
          </div>
          <div>
            <p className="text-xs text-foreground/60 mb-1">Clean Duration</p>
            <p className="text-2xl font-bold">{totalDuration}m</p>
          </div>
        </div>

        <div className="flex gap-2 pt-4 border-t border-border">
          <button
            onClick={handleApproveAll}
            className="flex-1 rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-card-muted"
          >
            Approve All
          </button>
          <button
            onClick={handleRejectAll}
            className="flex-1 rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-card-muted"
          >
            Reject All
          </button>
        </div>
      </div>

      {/* Segments List */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold">Audio Segments</h2>
        {mockSegments.map((segment, idx) => (
          <div
            key={segment.id}
            className="card-base flex items-center gap-4"
          >
            {/* Segment Number */}
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-card-muted flex-shrink-0">
              <span className="text-sm font-semibold">{String(idx + 1).padStart(2, "0")}</span>
            </div>

            {/* Waveform & Duration */}
            <div className="flex-1">
              <div className="mb-2 h-10 w-full rounded-lg bg-card-muted flex items-center justify-center">
                <div className="flex h-6 items-center gap-1">
                  {[...Array(15)].map((_, i) => (
                    <div
                      key={i}
                      className="w-0.5 bg-accent-amber rounded-full"
                      style={{
                        height: `${Math.random() * 20 + 10}px`,
                      }}
                    />
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs text-foreground/60">{segment.duration.toFixed(1)}s</p>
                <p className="text-xs text-foreground/40">
                  {segments[segment.id] ? "Approved" : "Rejected"}
                </p>
              </div>
            </div>

            {/* Player & Toggle */}
            <div className="flex items-center gap-2 flex-shrink-0">
              <button className="flex h-10 w-10 items-center justify-center rounded-lg bg-card-muted transition-colors hover:bg-accent-amber hover:text-background">
                <Play size={18} />
              </button>

              <button
                onClick={() => toggleSegment(segment.id)}
                className={`flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
                  segments[segment.id]
                    ? "bg-accent-emerald text-background"
                    : "bg-card-muted text-accent-rose"
                }`}
              >
                {segments[segment.id] ? (
                  <Check size={18} />
                ) : (
                  <X size={18} />
                )}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Info Box */}
      <div className="flex gap-3 items-start rounded-lg bg-card-muted p-4 border-l-4 border-accent-amber">
        <AlertCircle size={20} className="text-accent-amber flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-medium">Quality Guidelines</p>
          <ul className="list-disc list-inside text-foreground/60 mt-1 space-y-0.5">
            <li>Reject segments with background noise</li>
            <li>Reject very short clips (&lt;2s)</li>
            <li>Keep only clear, natural speech</li>
          </ul>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <button className="flex-1 rounded-lg border border-border px-6 py-3 font-medium transition-colors hover:bg-card-muted">
          Back
        </button>
        <button
          onClick={handleProceed}
          disabled={approvedCount === 0}
          className="flex-1 rounded-lg bg-accent-emerald px-6 py-3 font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Proceed to Hyperparameters ({approvedCount})
        </button>
      </div>
    </div>
  );
}
