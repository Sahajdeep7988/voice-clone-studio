"use client";

import { useState, useEffect } from "react";
import { Pause, Play, X, ChevronDown, ChevronUp } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { mockSessions } from "@/lib/mockData";

const mockLossHistory = [
  { epoch: 1, loss: 2.5 },
  { epoch: 10, loss: 2.1 },
  { epoch: 20, loss: 1.8 },
  { epoch: 30, loss: 1.5 },
  { epoch: 40, loss: 1.3 },
  { epoch: 42, loss: 1.2 },
];

const mockLogs = [
  "[RVC] epoch=40 step=1100 lowest_value=1.2504",
  "[RVC] epoch=41 step=1150 lowest_value=1.2104",
  "[RVC] epoch=42 step=1150 lowest_value=1.2045",
  "[Session] Updated session state to Supabase.",
  "[RVC] epoch=42 step=1200 g=0.0432 d=0.1023",
  "[Training] Model checkpoint saved: G_42e.pth",
];

export function ActiveTraining({ sessionId }: { sessionId: string }) {
  const session = mockSessions.find((s) => s.id === sessionId) || mockSessions[0];
  const [isExpanded, setIsExpanded] = useState(true);
  const [isPaused, setIsPaused] = useState(session.status === "paused");

  // Simulate live training
  const [currentEpoch, setCurrentEpoch] = useState(session.epochs.current);
  const [currentLoss, setCurrentLoss] = useState(session.loss);

  useEffect(() => {
    if (isPaused || session.status === "paused") return;

    const interval = setInterval(() => {
      setCurrentEpoch((prev) => {
        const next = prev + 1;
        return next > session.epochs.total ? session.epochs.total : next;
      });
      setCurrentLoss((prev) => Math.max(0.5, prev - Math.random() * 0.05));
    }, 2000);

    return () => clearInterval(interval);
  }, [isPaused, session.status, session.epochs.total]);

  const progress = (currentEpoch / session.epochs.total) * 100;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">{session.modelName}</h1>
        <p className="mt-2 text-foreground/60">
          {session.datasetDuration} minute dataset • {session.device}
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card-base">
          <p className="text-sm text-foreground/60 mb-2">Progress</p>
          <p className="text-3xl font-bold mb-3">
            {currentEpoch} / {session.epochs.total}
          </p>
          <div className="h-2 w-full overflow-hidden rounded-full bg-card-muted">
            <div
              className="h-full bg-accent-emerald transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="card-base">
          <p className="text-sm text-foreground/60 mb-2">Current Loss</p>
          <p className="text-3xl font-bold">{currentLoss.toFixed(4)}</p>
          <p className="text-xs text-accent-emerald mt-2">↘ Decreasing</p>
        </div>

        <div className="card-base">
          <p className="text-sm text-foreground/60 mb-2">Time Elapsed</p>
          <p className="text-3xl font-bold">~{Math.ceil((currentEpoch / 60) * 45)}m</p>
          <p className="text-xs text-foreground/50 mt-2">Estimated time remaining</p>
        </div>
      </div>

      {/* Loss Graph */}
      <div className="card-base">
        <h2 className="text-lg font-semibold mb-4">Loss Progression</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={mockLossHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid rgba(255,255,255,0.1)",
              }}
              labelStyle={{ color: "#fafafa" }}
            />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#10b981"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Controls */}
      <div className="flex gap-3">
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="flex items-center gap-2 rounded-lg bg-accent-violet px-6 py-3 font-medium text-background transition-opacity hover:opacity-90"
        >
          {isPaused ? (
            <>
              <Play size={20} />
              Resume
            </>
          ) : (
            <>
              <Pause size={20} />
              Pause
            </>
          )}
        </button>

        <button className="flex items-center gap-2 rounded-lg bg-accent-rose px-6 py-3 font-medium text-background transition-opacity hover:opacity-90">
          <X size={20} />
          Stop Training
        </button>
      </div>

      {/* Terminal Logs */}
      <div className="card-base space-y-3">
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <h2 className="text-lg font-semibold">Terminal Logs</h2>
          {isExpanded ? (
            <ChevronUp size={20} />
          ) : (
            <ChevronDown size={20} />
          )}
        </div>

        {isExpanded && (
          <div className="bg-background rounded-lg p-4 border border-border font-mono text-sm max-h-64 overflow-y-auto">
            {mockLogs.map((log, idx) => (
              <div key={idx} className="text-foreground/70">
                {log}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Checkpoints */}
      <div className="card-base space-y-3">
        <h2 className="text-lg font-semibold">Available Checkpoints</h2>
        <div className="space-y-2">
          {[42, 35, 28, 21].map((epoch) => (
            <div
              key={epoch}
              className="flex items-center justify-between rounded-lg bg-card-muted px-4 py-3"
            >
              <span className="text-sm font-medium">G_{epoch}e.pth</span>
              <button className="text-xs text-accent-amber hover:underline">
                Test Inference
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
