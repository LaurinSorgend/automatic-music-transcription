# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import pretty_midi

FRAME_RATE=100
N_PITCHES=88
PITCH_OFFSET=21

def base2midi(csv_path, data_root):
    df = pd.read_csv(csv_path)
    m={}
    for _,r in df.iterrows():
        b=os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        m[b]=os.path.normpath(os.path.join(data_root, r["midi_filename"]))
    return m

def compute_velocity_on(midi_path, T, fs=FRAME_RATE):
    """Velocity at onset frames only: (88, T) in [0,1]."""
    pm=pretty_midi.PrettyMIDI(midi_path)
    vel_on=np.zeros((N_PITCHES,T),np.float32)
    
    by = [[] for _ in range(N_PITCHES)]
    for inst in pm.instruments:
        for n in inst.notes:
            p = n.pitch - PITCH_OFFSET
            if 0 <= p < N_PITCHES:
                by[p].append(n)
            
    for p in range(N_PITCHES):
        by[p].sort(key=lambda n: n.start)
        for note in by[p]:
            s = int(round(note.start * fs))
            if s < T:
                vel_on[p, s] = note.velocity / 127.0
    return vel_on

def cc_timeline(pm, cc_num, T, fs=FRAME_RATE, thr=64):
    """
    Build a (T,) float32 in {0,1} for a CC lane using a threshold (default 64).
    Treats the CC as a step function over time.
    """
    t = np.zeros(T, np.float32)
    events = []
    for inst in pm.instruments:
        for c in inst.control_changes:
            if c.number == cc_num:
                events.append((c.time, c.value))
    if not events:
        return t
    events.sort(key=lambda x: x[0])

    i_prev = 0
    state = 1.0 if events[0][1] >= thr else 0.0
    for time, val in events:
        i = int(round(time * fs))
        i = max(0, min(T, i))
        t[i_prev:i] = state
        state = 1.0 if val >= thr else 0.0
        i_prev = i
    t[i_prev:T] = state
    return t

def compute_pedals(midi_path, T, fs = FRAME_RATE):
    pm = pretty_midi.PrettyMIDI(midi_path)
    ped64 = cc_timeline(pm, 64, T, fs)  # sustain
    ped66 = cc_timeline(pm, 66, T, fs)  # sostenuto
    ped67 = cc_timeline(pm, 67, T, fs)  # una corda
    # store as (1, T) to be consistent with other labels
    return ped64[None, :], ped66[None, :], ped67[None, :]

def upgrade_folder(src_npz, data_root, csv_path, dst_npz):
    os.makedirs(dst_npz, exist_ok=True)
    m = base2midi(csv_path, data_root)
    files = sorted(glob.glob(os.path.join(src_npz,"*.npz")))
    print("NPZs:", len(files))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        midi = m.get(base)
        if not midi or not os.path.exists(midi):
            print("skip (no MIDI):", base); continue

        d = np.load(f, mmap_mode="r")
        T = d["log_mel"].shape[1]
        
        vel_on = compute_velocity_on(midi, T)
        ped64, ped66, ped67 = compute_pedals(midi, T)
        
        out = os.path.join(dst_npz, base+".npz")
        np.savez_compressed(
            out,
            log_mel = d["log_mel"],
            frame = d["frame"], 
            onset=d["onset"], 
            offset = d["offset"],
            vel_on = vel_on.astype(np.float32),
            ped64=ped64.astype(np.float32),
            ped66=ped66.astype(np.float32),
            ped67=ped67.astype(np.float32)
        )
        print("wrote", out)

if __name__ == "__main__":
    ROOT = r"C:\Users\2406448\OneDrive - Abertay University\amt_project\data\maestro-v3.0.0"
    CSV = rf"{ROOT}\maestro-v3.0.0.csv"
    upgrade_folder(r"C:...\amt_project\preprocessed_full\train", ROOT, CSV,
                   r"C:...\amt_project\preprocessed_full_v3\train")
    upgrade_folder(r"C:...\amt_project\preprocessed_full\val", ROOT, CSV,
                   r"C:...\amt_project\preprocessed_full_v3\val")
    upgrade_folder(r"C:...\amt_project\preprocessed_full\test", ROOT, CSV,
                   r"C:...\amt_project\preprocessed_full_v3\test")

