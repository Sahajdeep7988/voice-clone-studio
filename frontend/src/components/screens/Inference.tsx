"use client";

import { useState } from "react";
import { Upload, Play, Download, AlertCircle } from "lucide-react";
import { mockModels } from "@/lib/mockData";

interface InferenceState {
  selectedModel: string;
  sourceAudio: File | null;
  pitchShift: number;
  isProcessing: boolean;
}

export function Inference() {
  const [state, setState] = useState<InferenceState>({
    selectedModel: mockModels.find((m) => m.status === "done")?.id || mockModels[0].id,
    sourceAudio: null,
    pitchShift: 0,
    isProcessing: false,
  });

  const selectedModel = mockModels.find((m) => m.id === state.selectedModel);
  const isModelLocal = selectedModel?.isLocal ?? false;

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setState((prev) => ({ ...prev, sourceAudio: e.target.files![0] }));
    }
  };

  const handleConvert = async () => {
    setState((prev) => ({ ...prev, isProcessing: true }));
    
    // Simulate processing
    setTimeout(() => {
      console.log("[v0] Conversion complete:", {
        model: state.selectedModel,
        sourceAudio: state.sourceAudio?.name,
        pitchShift: state.pitchShift,
      });
      setState((prev) => ({ ...prev, isProcessing: false }));
      alert("Conversion complete! Output saved to ~/VoiceClone/output/");
    }, 3000);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Voice Conversion</h1>
        <p className="mt-2 text-foreground/60">
          Convert any audio to your cloned voice using a trained model
        </p>
      </div>

      {/* Model Selection */}
      <div className="card-base space-y-3">
        <label htmlFor="model-select" className="block text-sm font-medium">
          Select Voice Model
        </label>
        <select
          id="model-select"
          value={state.selectedModel}
          onChange={(e) => setState((prev) => ({ ...prev, selectedModel: e.target.value }))}
          className="w-full rounded-lg bg-card-muted px-4 py-3 border border-border focus:outline-none focus:border-accent-amber transition-colors"
        >
          {mockModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.status})
            </option>
          ))}
        </select>

        {!isModelLocal && (
          <div className="flex gap-2 items-start rounded-lg bg-card-muted p-3 border-l-4 border-accent-rose">
            <AlertCircle size={16} className="text-accent-rose flex-shrink-0 mt-0.5" />
            <p className="text-xs text-foreground/70">
              This model is not available locally. Download it first from My Models.
            </p>
          </div>
        )}
      </div>

      {/* Source Audio Upload */}
      <div className="card-base space-y-4">
        <label className="block text-sm font-medium">Source Audio</label>
        <div className="rounded-lg border-2 border-dashed border-border p-8 text-center">
          <Upload className="mx-auto mb-3 text-foreground/40" size={32} />
          <h3 className="font-semibold">Upload Audio File</h3>
          <p className="text-sm text-foreground/60 mt-1">
            Supported: MP3, WAV, FLAC, MP4
          </p>
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={handleAudioChange}
            className="hidden"
            id="source-audio"
          />
          <label
            htmlFor="source-audio"
            className="mt-4 inline-block rounded-lg bg-accent-amber px-6 py-2 font-medium text-background cursor-pointer hover:opacity-90"
          >
            Choose File
          </label>
        </div>

        {state.sourceAudio && (
          <div className="rounded-lg bg-card-muted px-4 py-3">
            <p className="text-sm font-medium">{state.sourceAudio.name}</p>
            <p className="text-xs text-foreground/60 mt-1">
              {(state.sourceAudio.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}
      </div>

      {/* Pitch Shift */}
      <div className="card-base space-y-4">
        <div>
          <label htmlFor="pitch" className="block text-sm font-medium mb-2">
            Pitch Shift (semitones)
          </label>
          <div className="flex items-center gap-4">
            <input
              id="pitch"
              type="range"
              min="-12"
              max="12"
              value={state.pitchShift}
              onChange={(e) =>
                setState((prev) => ({ ...prev, pitchShift: parseInt(e.target.value) }))
              }
              className="flex-1 h-2 bg-card-muted rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-lg font-semibold w-12 text-right">
              {state.pitchShift > 0 ? "+" : ""}{state.pitchShift}
            </span>
          </div>
          <p className="text-xs text-foreground/60 mt-2">
            -12 (lower) to +12 (higher)
          </p>
        </div>
      </div>

      {/* Advanced Options */}
      <details className="card-base">
        <summary className="cursor-pointer font-medium flex items-center justify-between">
          Advanced Options
          <span>▼</span>
        </summary>
        <div className="mt-4 space-y-4 pt-4 border-t border-border">
          <div>
            <label htmlFor="index-rate" className="block text-sm font-medium mb-2">
              Index Rate (0-1)
            </label>
            <input
              id="index-rate"
              type="range"
              min="0"
              max="1"
              step="0.1"
              defaultValue="0.5"
              className="w-full"
            />
            <p className="text-xs text-foreground/60 mt-1">
              Higher = more accurate to original voice
            </p>
          </div>

          <div>
            <label htmlFor="protect" className="block text-sm font-medium mb-2">
              Protect Breathiness (0-1)
            </label>
            <input
              id="protect"
              type="range"
              min="0"
              max="1"
              step="0.1"
              defaultValue="0.5"
              className="w-full"
            />
          </div>
        </div>
      </details>

      {/* Convert Button */}
      <button
        onClick={handleConvert}
        disabled={!state.sourceAudio || !isModelLocal || state.isProcessing}
        className="w-full flex items-center justify-center gap-2 rounded-lg bg-accent-emerald px-6 py-4 font-semibold text-background transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {state.isProcessing ? (
          <>
            <span className="animate-spin">⌛</span>
            Converting... (Est. 30s)
          </>
        ) : (
          <>
            <Play size={20} />
            Convert to Voice
          </>
        )}
      </button>

      {/* Output Preview */}
      <div className="card-base space-y-4">
        <h2 className="text-lg font-semibold">Converted Audio</h2>
        <div className="rounded-lg bg-card-muted p-6 text-center">
          <p className="text-foreground/60 mb-4">No audio generated yet</p>
          <div className="flex gap-2 justify-center">
            <button
              disabled
              className="flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              <Play size={16} />
              Play
            </button>
            <button
              disabled
              className="flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium disabled:opacity-50"
            >
              <Download size={16} />
              Download
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
