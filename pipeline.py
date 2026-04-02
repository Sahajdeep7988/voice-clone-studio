#!/usr/bin/env python3
"""
Voice Clone Studio — CLI Pipeline Runner
Connects all modules end-to-end for testing the full pipeline.

Usage:
    python pipeline.py --files audio1.mp3 audio2.wav --model-name myvoice
    python pipeline.py --files audio.mp3 --model-name myvoice \
                       --email user@example.com --password secret
"""

import argparse
import sys
import time
from datetime import datetime, timezone


def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Voice Clone Studio — local voice cloning pipeline"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Input audio/video files (MP3, WAV, FLAC, MP4, MKV)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        metavar="NAME",
        help="Name for the trained voice model",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Supabase account email (optional)",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Supabase account password (optional)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — AUDIO PREPROCESSING")
    print("=" * 60)

    try:
        from core.preprocessing import AudioPreprocessor, PreprocessingError
        preprocessor = AudioPreprocessor()
        stats = preprocessor.process(args.files)
    except PreprocessingError as e:
        print(f"\n[ERROR] Preprocessing failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected preprocessing error: {e}")
        sys.exit(1)

    print("\n--- Quality Report ---")
    print(f"  Segments:       {stats['segment_count']}")
    print(f"  Total duration: {stats['total_duration_minutes']:.2f} minutes")
    print(f"  Quality label:  {stats['quality_label']}")
    print(f"  Segment paths:  {len(stats['segment_paths'])} files in dataset/segments/")

    # ------------------------------------------------------------------
    # Step 2: LLM Hyperparameter Config
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — LLM HYPERPARAMETER CONFIG")
    print("=" * 60)

    try:
        from core.llm_config import LLMConfigurator
        configurator = LLMConfigurator()
        hyperparams = configurator.get_hyperparameters(stats)
    except Exception as e:
        print(f"\n[ERROR] Hyperparameter config failed: {e}")
        sys.exit(1)

    print("\n--- Hyperparameters ---")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Step 3: Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — RVC TRAINING")
    print("=" * 60)
    print(f"  Model name:   {args.model_name}")
    print(f"  Dataset path: dataset/segments")
    print()

    epochs_completed = 0

    def on_progress(epoch: int, loss: float):
        nonlocal epochs_completed
        epochs_completed = epoch
        print(f"  Epoch {epoch:>4d} | Loss {loss:.6f}")

    try:
        from core.training_engine import TrainingEngine
        engine = TrainingEngine()
        model_path = engine.start_training(
            dataset_path="dataset/segments",
            model_name=args.model_name,
            hyperparams=hyperparams,
            progress_callback=on_progress,
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

    print(f"\n[Pipeline] Model saved: {model_path}")

    # ------------------------------------------------------------------
    # Step 4: Supabase sync (optional)
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time

    if args.email and args.password:
        print("\n" + "=" * 60)
        print("STEP 4 — SUPABASE SYNC")
        print("=" * 60)

        try:
            from services.supabase_client import SupabaseClient
            from core.llm_config import LLMConfigurator

            client = SupabaseClient()
            logged_in = client.login(args.email, args.password)

            if logged_in:
                hw_info = LLMConfigurator()._scan_hardware()
                metadata = {
                    "user_id":          client.user_id,
                    "model_name":       args.model_name,
                    "status":           "completed",
                    "duration_seconds": int(elapsed),
                    "epochs_completed": epochs_completed,
                    "hardware_profile": hw_info,
                    "created_at":       datetime.now(timezone.utc).isoformat(),
                }
                synced = client.sync_training_session(metadata)
                if not synced:
                    print("[Pipeline] Session sync failed (continuing anyway).")
            else:
                print("[Pipeline] Login failed — skipping sync.")
        except Exception as e:
            print(f"[Pipeline] Supabase step error (non-fatal): {e}")
    else:
        print("\n[Pipeline] No credentials provided — skipping Supabase sync.")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Quality:  {stats['quality_label']}")
    print()


if __name__ == "__main__":
    run_pipeline()
