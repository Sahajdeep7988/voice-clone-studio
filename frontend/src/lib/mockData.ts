export interface TrainingSession {
  id: string;
  modelName: string;
  status: "created" | "preprocessing" | "training" | "paused" | "done" | "error";
  epochs: { current: number; total: number };
  loss: number;
  datasetDuration: number; // in minutes
  device: string;
  createdAt: string;
  updatedAt: string;
  checkpoint?: string;
}

export interface VoiceModel {
  id: string;
  name: string;
  status: "done" | "paused" | "training";
  epochs: { current: number; total: number };
  duration: number; // in minutes
  device: string;
  isLocal: boolean;
}

export interface AudioSegment {
  id: string;
  duration: number;
  approved: boolean;
  waveformUrl: string;
}

// Mock Sessions
export const mockSessions: TrainingSession[] = [
  {
    id: "session-1",
    modelName: "Podcast Voice",
    status: "done",
    epochs: { current: 250, total: 250 },
    loss: 0.1234,
    datasetDuration: 14,
    device: "LOCAL",
    createdAt: "2024-04-01T10:00:00Z",
    updatedAt: "2024-04-02T15:30:00Z",
    checkpoint: "G_250e.pth",
  },
  {
    id: "session-2",
    modelName: "Gaming Alter Ego",
    status: "paused",
    epochs: { current: 100, total: 150 },
    loss: 0.3456,
    datasetDuration: 8,
    device: "MACBOOK-PRO",
    createdAt: "2024-04-03T14:00:00Z",
    updatedAt: "2024-04-03T16:45:00Z",
  },
  {
    id: "session-3",
    modelName: "Deep Voice",
    status: "training",
    epochs: { current: 42, total: 200 },
    loss: 0.5678,
    datasetDuration: 12,
    device: "LOCAL",
    createdAt: "2024-04-04T08:00:00Z",
    updatedAt: "2024-04-04T14:20:00Z",
  },
];

// Mock Models
export const mockModels: VoiceModel[] = [
  {
    id: "model-1",
    name: "Podcast Voice",
    status: "done",
    epochs: { current: 250, total: 250 },
    duration: 14,
    device: "LOCAL",
    isLocal: true,
  },
  {
    id: "model-2",
    name: "Gaming Alter Ego",
    status: "paused",
    epochs: { current: 100, total: 150 },
    duration: 8,
    device: "MACBOOK-PRO",
    isLocal: false,
  },
  {
    id: "model-3",
    name: "Deep Voice",
    status: "training",
    epochs: { current: 42, total: 200 },
    duration: 12,
    device: "LOCAL",
    isLocal: true,
  },
  {
    id: "model-4",
    name: "Whisper Voice",
    status: "done",
    epochs: { current: 180, total: 180 },
    duration: 10,
    device: "LOCAL",
    isLocal: true,
  },
];

// Mock Audio Segments
export const mockSegments: AudioSegment[] = [
  { id: "seg-1", duration: 4.2, approved: true, waveformUrl: "/waveform-1.svg" },
  { id: "seg-2", duration: 1.8, approved: false, waveformUrl: "/waveform-2.svg" },
  { id: "seg-3", duration: 5.5, approved: true, waveformUrl: "/waveform-3.svg" },
  { id: "seg-4", duration: 3.1, approved: true, waveformUrl: "/waveform-4.svg" },
  { id: "seg-5", duration: 6.8, approved: true, waveformUrl: "/waveform-5.svg" },
];

// System Stats
export const mockSystemStats = {
  gpuVram: "24 GB (NVIDIA RTX 4090)",
  gpuStatus: "✓ CUDA Available",
  torchVersion: "2.7.1",
  llmModel: "gemma-2b-q4.gguf",
  totalModels: 4,
  totalSessions: 3,
};
