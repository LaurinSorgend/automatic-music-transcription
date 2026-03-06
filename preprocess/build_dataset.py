# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import pretty_midi
import pandas as pd

# Feature extraction 
def compute_log_mel(wav_path, sr=16000, n_mels=128, hop_length=160, n_fft=1024, win_length=1024):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    mu = log_S.mean(axis=1, keepdims=True)
    sg = log_S.std(axis=1, keepdims=True) + 1e-6
    return ((log_S - mu) / sg).astype(np.float32)
 
# Midi targetting
def midi_to_targets(midi_path, frame_rate=100, n_pitches=88, pitch_offset=21):
    pm = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = pm.get_piano_roll(fs=frame_rate)[pitch_offset:pitch_offset+n_pitches]
    frame  = (piano_roll > 0).astype(np.float32)
    onset  = np.zeros_like(frame, dtype=np.float32)
    offset = np.zeros_like(frame, dtype=np.float32)
    for inst in pm.instruments:
        for note in inst.notes:
            p = note.pitch - pitch_offset
            if 0 <= p < n_pitches:
                start = int(round(note.start * frame_rate))
                end   = int(round(note.end   * frame_rate))
                if start < onset.shape[1]:
                    onset[p, start] = 1.0
                if end < offset.shape[1]:
                    offset[p, end] = 1.0
    return frame, onset, offset

# Data split
def build_split(split_name, data_root, out_dir, sr=16000, frame_rate=100):
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(data_root, "maestro-v3.0.0.csv"))
    df = meta[meta["split"] == split_name]

    for _, row in df.iterrows():
        wav_path = os.path.normpath(os.path.join(data_root, row["audio_filename"]))
        midi_path = os.path.normpath(os.path.join(data_root, row["midi_filename"]))
        base = os.path.splitext(os.path.basename(wav_path))[0]
        if not (os.path.exists(wav_path) and os.path.exists(midi_path)):
            print(f"skip (missing): {base}")
            continue

        print(f"[{split_name}] {base}")
        log_mel = compute_log_mel(wav_path, sr=sr)
        frame, onset, offset = midi_to_targets(midi_path, frame_rate=frame_rate)

        # Align lengths
        T = min(log_mel.shape[1], frame.shape[1])
        log_mel = log_mel[:, :T]
        frame = frame[:, :T]
        onset = onset[:, :T]
        offset = offset[:, :T]

        out_path = os.path.join(out_dir, f"{base}.npz") 
        np.savez_compressed(out_path,
            log_mel=log_mel.astype(np.float32),
            frame=frame.astype(np.float32),
            onset=onset.astype(np.float32),
            offset=offset.astype(np.float32)
        )
        print("  saved to", out_path)

if __name__ == "__main__":
    ROOT = r"C:...\amt_project\data\maestro-v3.0.0"
    build_split("validation", ROOT, r"C:...\amt_project\preprocessed\val")
    build_split("train", ROOT, r"C:...\preprocessed_full\train")
    build_split("test", ROOT, r"C:...\preprocessed_full\test")