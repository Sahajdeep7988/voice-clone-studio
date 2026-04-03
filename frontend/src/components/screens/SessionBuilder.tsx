"use client";

import { useState } from "react";
import { Upload, AlertCircle } from "lucide-react";

interface FormState {
  modelName: string;
  advancedMode: boolean;
  files: File[];
}

export function SessionBuilder() {
  const [form, setForm] = useState<FormState>({
    modelName: "",
    advancedMode: false,
    files: [],
  });

  const [step, setStep] = useState<"files" | "config" | "review">("files");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setForm((prev) => ({
        ...prev,
        files: Array.from(e.target.files!),
      }));
    }
  };

  const handleNext = () => {
    if (step === "files" && form.files.length > 0) {
      setStep("config");
    } else if (step === "config" && form.modelName.trim()) {
      setStep("review");
    }
  };

  const handleSubmit = () => {
    console.log("[v0] Session created:", form);
    alert("Session created! Check console for details.");
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Create Training Session</h1>
        <p className="mt-2 text-foreground/60">
          {step === "files" && "Upload audio or video files for voice cloning"}
          {step === "config" && "Configure your training model"}
          {step === "review" && "Review and start training"}
        </p>
      </div>

      {/* Progress Indicator */}
      <div className="flex gap-2">
        {["files", "config", "review"].map((s, idx) => (
          <div
            key={s}
            className={`h-1 flex-1 rounded-full transition-colors ${
              step === s || ["files", "config", "review"].indexOf(step) >= idx
                ? "bg-accent-amber"
                : "bg-card-muted"
            }`}
          />
        ))}
      </div>

      {/* Step: Files */}
      {step === "files" && (
        <div className="space-y-6">
          <div className="card-base flex flex-col gap-6">
            <div className="rounded-lg border-2 border-dashed border-border p-12 text-center">
              <Upload className="mx-auto mb-4 text-foreground/40" size={32} />
              <h3 className="font-semibold">Upload Audio or Video Files</h3>
              <p className="mt-2 text-sm text-foreground/60">
                Supported formats: MP3, WAV, FLAC, MP4, MKV
              </p>
              <input
                type="file"
                multiple
                accept="audio/*,video/*"
                onChange={handleFileChange}
                className="mt-4 hidden"
                id="file-input"
              />
              <label
                htmlFor="file-input"
                className="mt-4 inline-block rounded-lg bg-accent-amber px-6 py-2 font-medium text-background cursor-pointer hover:opacity-90"
              >
                Select Files
              </label>
            </div>

            {form.files.length > 0 && (
              <div>
                <h4 className="mb-3 font-semibold">Selected Files ({form.files.length})</h4>
                <ul className="space-y-2">
                  {form.files.map((file, idx) => (
                    <li
                      key={idx}
                      className="flex items-center justify-between rounded-lg bg-card-muted px-4 py-2"
                    >
                      <span className="text-sm">{file.name}</span>
                      <span className="text-xs text-foreground/60">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex gap-3 items-center rounded-lg bg-card-muted p-4 border-l-4 border-accent-amber">
              <AlertCircle size={20} className="text-accent-amber flex-shrink-0" />
              <div className="text-sm">
                <p className="font-medium">Minimum 5 minutes of clean audio required</p>
                <p className="text-foreground/60">10-20 minutes recommended for best quality</p>
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3">
            <button
              onClick={() => setStep("config")}
              disabled={form.files.length === 0}
              className="rounded-lg border border-border px-6 py-3 font-medium transition-colors hover:bg-card-muted disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Skip (Use All Files)
            </button>
            <button
              onClick={handleNext}
              disabled={form.files.length === 0}
              className="rounded-lg bg-accent-amber px-6 py-3 font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step: Configuration */}
      {step === "config" && (
        <div className="space-y-6">
          <div className="card-base space-y-4">
            <div>
              <label htmlFor="model-name" className="block text-sm font-medium mb-2">
                Model Name
              </label>
              <input
                id="model-name"
                type="text"
                placeholder="e.g., Podcast Voice, Gaming Alter Ego"
                value={form.modelName}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, modelName: e.target.value }))
                }
                className="w-full rounded-lg bg-card-muted px-4 py-3 border border-border focus:outline-none focus:border-accent-amber transition-colors"
              />
            </div>

            <div>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={form.advancedMode}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      advancedMode: e.target.checked,
                    }))
                  }
                  className="w-5 h-5 rounded cursor-pointer"
                />
                <div>
                  <p className="font-medium">Advanced Mode</p>
                  <p className="text-sm text-foreground/60">
                    Review and approve audio segments, customize hyperparameters
                  </p>
                </div>
              </label>
            </div>
          </div>

          <div className="flex justify-between gap-3">
            <button
              onClick={() => setStep("files")}
              className="rounded-lg border border-border px-6 py-3 font-medium transition-colors hover:bg-card-muted"
            >
              Back
            </button>
            <button
              onClick={handleNext}
              disabled={!form.modelName.trim()}
              className="rounded-lg bg-accent-amber px-6 py-3 font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Review
            </button>
          </div>
        </div>
      )}

      {/* Step: Review */}
      {step === "review" && (
        <div className="space-y-6">
          <div className="card-base space-y-4">
            <div>
              <p className="text-sm text-foreground/60 mb-1">Model Name</p>
              <p className="text-lg font-semibold">{form.modelName}</p>
            </div>
            <div className="border-t border-border pt-4">
              <p className="text-sm text-foreground/60 mb-2">Files ({form.files.length})</p>
              <ul className="space-y-2">
                {form.files.map((file, idx) => (
                  <li key={idx} className="text-sm">
                    {file.name}
                  </li>
                ))}
              </ul>
            </div>
            <div className="border-t border-border pt-4">
              <p className="text-sm text-foreground/60 mb-1">Mode</p>
              <p className="text-sm">
                {form.advancedMode ? "Advanced (with curation)" : "Simple (fire-and-forget)"}
              </p>
            </div>
          </div>

          <div className="flex justify-between gap-3">
            <button
              onClick={() => setStep("config")}
              className="rounded-lg border border-border px-6 py-3 font-medium transition-colors hover:bg-card-muted"
            >
              Back
            </button>
            <button
              onClick={handleSubmit}
              className="rounded-lg bg-accent-emerald px-6 py-3 font-medium text-background transition-opacity hover:opacity-90"
            >
              Start Training
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
