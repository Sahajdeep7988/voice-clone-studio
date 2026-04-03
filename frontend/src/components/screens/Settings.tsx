"use client";

import { useState } from "react";
import { ChevronRight, ToggleRight } from "lucide-react";
import { mockSystemStats } from "@/lib/mockData";

export function Settings() {
  const [supabaseEmail, setSupabaseEmail] = useState("");
  const [syncEnabled, setSyncEnabled] = useState(false);

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="mt-2 text-foreground/60">
          Configure your Voice Clone Studio preferences and integrations
        </p>
      </div>

      {/* System Information */}
      <section>
        <h2 className="text-xl font-semibold mb-4">System Information</h2>
        <div className="card-base space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <span className="text-foreground/60">GPU / VRAM</span>
            <span className="font-semibold">{mockSystemStats.gpuVram}</span>
          </div>
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <span className="text-foreground/60">CUDA Status</span>
            <span className="font-semibold text-accent-emerald">{mockSystemStats.gpuStatus}</span>
          </div>
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <span className="text-foreground/60">PyTorch Version</span>
            <span className="font-semibold">{mockSystemStats.torchVersion}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-foreground/60">LLM Model</span>
            <span className="font-semibold text-sm">{mockSystemStats.llmModel}</span>
          </div>
        </div>
      </section>

      {/* Cloud Sync */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Cloud Sync (Supabase)</h2>
        <div className="card-base space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Enable Cloud Sync</p>
              <p className="text-sm text-foreground/60 mt-1">
                Sync training sessions across devices and machines
              </p>
            </div>
            <button
              onClick={() => setSyncEnabled(!syncEnabled)}
              className={`relative h-8 w-14 rounded-full transition-colors ${
                syncEnabled ? "bg-accent-emerald" : "bg-card-muted"
              }`}
            >
              <div
                className={`absolute top-1 h-6 w-6 rounded-full bg-white transition-transform ${
                  syncEnabled ? "translate-x-7" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          {syncEnabled && (
            <div className="space-y-3 pt-4 border-t border-border">
              <div>
                <label htmlFor="email" className="block text-sm font-medium mb-2">
                  Supabase Email
                </label>
                <input
                  id="email"
                  type="email"
                  placeholder="your@email.com"
                  value={supabaseEmail}
                  onChange={(e) => setSupabaseEmail(e.target.value)}
                  className="w-full rounded-lg bg-card-muted px-4 py-3 border border-border focus:outline-none focus:border-accent-amber transition-colors"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium mb-2">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  className="w-full rounded-lg bg-card-muted px-4 py-3 border border-border focus:outline-none focus:border-accent-amber transition-colors"
                />
              </div>

              <button className="w-full rounded-lg bg-accent-amber px-4 py-3 font-medium text-background transition-opacity hover:opacity-90">
                Connect to Supabase
              </button>

              <p className="text-xs text-foreground/60">
                Your credentials are stored locally and never transmitted to our servers.
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Training Preferences */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Training Preferences</h2>
        <div className="card-base space-y-3">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div>
              <p className="font-medium">Auto-save Checkpoints</p>
              <p className="text-xs text-foreground/60">Save model every 10 epochs</p>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5 rounded cursor-pointer" />
          </div>

          <div className="flex items-center justify-between pb-3 border-b border-border">
            <div>
              <p className="font-medium">Notify on Completion</p>
              <p className="text-xs text-foreground/60">Show alerts when training finishes</p>
            </div>
            <input type="checkbox" defaultChecked className="w-5 h-5 rounded cursor-pointer" />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">Keep Temporary Files</p>
              <p className="text-xs text-foreground/60">Preserve intermediate data for debugging</p>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded cursor-pointer" />
          </div>
        </div>
      </section>

      {/* Storage & Cache */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Storage & Cache</h2>
        <div className="card-base space-y-4">
          <div>
            <p className="font-medium">Cache Location</p>
            <p className="text-sm text-foreground/60 mt-1">~/.voice_clone_studio</p>
          </div>

          <div className="pt-4 border-t border-border">
            <p className="font-medium mb-3">Disk Usage</p>
            <div className="space-y-2">
              {[
                { label: "Models", size: "8.5 GB", percentage: 85 },
                { label: "Datasets", size: "2.1 GB", percentage: 45 },
                { label: "Logs", size: "512 MB", percentage: 20 },
              ].map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">{item.label}</span>
                    <span className="text-xs text-foreground/60">{item.size}</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-card-muted">
                    <div
                      className="h-full bg-accent-amber"
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <button className="text-sm text-accent-rose hover:underline">
              Clear Cache & Temporary Files
            </button>
          </div>
        </div>
      </section>

      {/* About */}
      <section>
        <h2 className="text-xl font-semibold mb-4">About</h2>
        <div className="card-base space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <span className="text-foreground/60">App Version</span>
            <span className="font-semibold">0.1.0</span>
          </div>
          <div className="flex items-center justify-between pb-3 border-b border-border">
            <span className="text-foreground/60">RVC Version</span>
            <span className="font-semibold">Latest (Applio)</span>
          </div>

          <div className="pt-2">
            <p className="text-xs text-foreground/60">
              Voice Clone Studio is a production-ready, local-first voice cloning pipeline built on
              Retrieval-based Voice Conversion (RVC).
            </p>
          </div>

          <div className="flex gap-2 pt-4">
            <button className="flex-1 flex items-center justify-between rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-card-muted">
              Documentation
              <ChevronRight size={16} />
            </button>
            <button className="flex-1 flex items-center justify-between rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-card-muted">
              GitHub
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
