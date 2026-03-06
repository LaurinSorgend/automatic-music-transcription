# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
import pretty_midi

from model.velped_conformer_amt import AMTConformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_MELS = 128
FRAME_RATE = 100    
SEG_FRAMES = 800    
HOP_FRAMES = 200    

TH_ON = 0.08
TH_FR = 0.26
TH_OFF = 0.22
TH_PED = 0.50
 # drop short notes
MIN_DUR_SEC = 0.06   

def load_model(ckpt_path: str) -> AMTConformer:
    model = AMTConformer(n_mels=N_MELS).to(DEVICE).eval()
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    return model

@torch.no_grad()
def infer_full_track(npz_path: str, model: AMTConformer):
    d = np.load(npz_path, mmap_mode="r")
    mel = d["log_mel"]          
    T_full = mel.shape[1]

    # (T_full, 88) then transpose at end
    on_sum = np.zeros((T_full, 88), dtype=np.float32)
    fr_sum = np.zeros((T_full, 88), dtype=np.float32)
    off_sum = np.zeros((T_full, 88), dtype=np.float32)
    vel_sum = np.zeros((T_full, 88), dtype=np.float32)
    p64_sum = np.zeros((T_full, ), dtype=np.float32)
    p67_sum = np.zeros((T_full, ), dtype=np.float32)   
    cnt = np.zeros((T_full, 1), dtype=np.float32)

    t0 = 0
    while t0 < T_full:
        t1 = min(t0 + SEG_FRAMES, T_full)
        seg = mel[:, t0:t1]
        valid = seg.shape[1]
        if valid < SEG_FRAMES:
            seg = np.pad(seg, ((0,0),(0, SEG_FRAMES - valid)), mode="constant")

        x = torch.from_numpy(seg).unsqueeze(0).float().to(DEVICE)  
        on_logit, fr_logit, off_logit, vel_logit, ped64_logit, ped67_logit = model(x)
                   # (1, SEG, 88)
        on  = torch.sigmoid(on_logit).squeeze(0).cpu().numpy()[:valid]    
        fr  = torch.sigmoid(fr_logit).squeeze(0).cpu().numpy()[:valid]
        off = torch.sigmoid(off_logit).squeeze(0).cpu().numpy()[:valid]
        vel = torch.sigmoid(vel_logit).squeeze(0).cpu().numpy()[:valid]
        p64 = torch.sigmoid(ped64_logit).squeeze(0).squeeze(-1).cpu().numpy()[:valid]  
        p67 = torch.sigmoid(ped67_logit).squeeze(0).squeeze(-1).cpu().numpy()[:valid]
        
        on_sum [t0:t1] += on
        fr_sum [t0:t1] += fr
        off_sum[t0:t1] += off
        vel_sum[t0:t1] += vel
        p64_sum[t0:t1] += p64
        p67_sum[t0:t1] += p67        
        cnt    [t0:t1] += 1.0

        t0 += HOP_FRAMES

    div = np.clip(cnt, 1e-6, None)  
    on_prob = (on_sum  / div).T   
    fr_prob = (fr_sum  / div).T
    off_prob = (off_sum / div).T
    vel_prob = (vel_sum / div).T
    ped64 = (p64_sum / div.squeeze(-1))  
    ped67 = (p67_sum / div.squeeze(-1))
    return on_prob, fr_prob, off_prob, vel_prob, ped64, ped67

def smooth_vec(x, win=5):
    if x is None or win <= 1: 
        return x
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=np.float32) / win
    y = np.convolve(xp, k, mode="valid")
    return y[:len(x)]

def decode_pedal(prob01, th_on=0.55, th_off=0.45, hop_s=0.01,
                 smooth_win=5, min_on_ms=80, min_off_ms=60):
    if prob01 is None:
        return None, None, None

    # smooth
    p = smooth_vec(prob01, smooth_win).astype(np.float32)

    # hysteresis
    state = np.zeros_like(p, dtype=np.uint8)
    on = False
    for t, v in enumerate(p):
        if not on and v >= th_on:
            on = True
        elif on and v <= th_off:
            on = False
        state[t] = 1 if on else 0

    # run-length helpers
    def _runs(arr):
        idx = np.flatnonzero(np.diff(np.concatenate(([0], arr, [0]))))
        starts, ends = idx[::2], idx[1::2]
        return list(zip(starts, ends))  # [start,end)

    hop_ms = hop_s * 1000.0

    # drop short ONs
    runs = [(a, b) for (a, b) in _runs(state) if (b - a) * hop_ms >= min_on_ms]

    # merge short OFF gaps
    merged, i = [], 0
    while i < len(runs):
        a, b = runs[i]
        while i + 1 < len(runs) and (runs[i + 1][0] - b) * hop_ms < min_off_ms:
            b = runs[i + 1][1]
            i += 1
        merged.append((a, b))
        i += 1

    state2 = np.zeros_like(state)
    for a, b in merged:
        state2[a:b] = 1

    downs = np.where((state2[1:] == 1) & (state2[:-1] == 0))[0] + 1
    ups = np.where((state2[1:] == 0) & (state2[:-1] == 1))[0] + 1
    return state2, downs, ups
        
def decode_to_midi(on, fr, off, out_path, frame_rate=FRAME_RATE, default_vel=64, vel=None,
                   ped64=None, ped67=None, use_pedals_for_timing=True):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand

    # smoothing for time axes 
    def smooth_time(prob, width=3):
        if width <= 1:
            return prob
        pad = width // 2
        T = prob.shape[1]
        out = np.zeros_like(prob)
        for p in range(prob.shape[0]):
            x = np.pad(prob[p], (pad, pad), mode="edge")
            c = np.convolve(x, np.ones(width)/width, mode="valid")
            out[p, :T] = c[:T]
        return out

    # smooth for stability
    on_s  = smooth_time(on, 3)
    fr_s  = smooth_time(fr, 5)
    off_s = smooth_time(off, 3)

    # pedal decoding (hysteresis + debounce) 
    hop_s = 1.0 / frame_rate
    ped64_state, ped64_downs, ped64_ups = decode_pedal(
        ped64, th_on=0.55, th_off=0.45, hop_s=hop_s, smooth_win=5, min_on_ms=80, min_off_ms=60
    )
    ped67_state, ped67_downs, ped67_ups = decode_pedal(
        ped67, th_on=0.55, th_off=0.45, hop_s=hop_s, smooth_win=5, min_on_ms=80, min_off_ms=60
    )

    def write_cc_lane(number, state):
        if state is None:
            return
        prev = -1
        for t, v in enumerate(state):
            if v != prev:
                inst.control_changes.append(pretty_midi.ControlChange(
                    number=number, value=127 if v == 1 else 0, time=t / frame_rate
                ))
                prev = v

    write_cc_lane(64, ped64_state)  # sustain
    write_cc_lane(67, ped67_state)  # una corda

    # Frame hysteresis
    FR_ON  = TH_FR                      # e.g., 0.26
    FR_OFF = max(TH_FR - 0.06, 0.05)    # e.g., 0.20

    sustain_frames = 3   
    refractory_fr  = 5    
    last_end_t = np.full(88, -9999, dtype=int)

    T = on.shape[1]
    s64 = ped64_state
    s64_prev = None if s64 is None else np.concatenate(([0], s64[:-1]))

    for p in range(88):
        active = False
        start = 0
        start_vel = int(np.clip(default_vel, 1, 127))

        t = 0
        while t < T:
            # thresholds with hysteresis for frame
            is_on = on_s[p, t]  > TH_ON
            fr_on_now = fr_s[p, t]  > FR_ON
            fr_off_now = fr_s[p, t]  <= FR_OFF
            is_off = off_s[p, t] > TH_OFF

            # pedal state
            s64_on = 0 if s64 is None else s64[t]
            s64_up = 0 if (s64 is None or s64_prev is None) else (s64_prev[t] == 1 and s64[t] == 0)

            start_ok = (is_on and fr_on_now)
            if (not active) and start_ok and (t - last_end_t[p] >= refractory_fr):
                if t + sustain_frames <= T and (fr_s[p, t:t+sustain_frames] > FR_ON).all():
                    # velocity from raw onset peak
                    if vel is not None:
                        t0, t1 = t, min(t + 5, T)
                        peak = t0 + int(np.argmax(on[p, t0:t1]))
                        start_vel = int(np.clip(vel[p, peak] * 127.0, 1, 127))
                    else:
                        start_vel = int(np.clip(default_vel, 1, 127))
                    active = True
                    start = t

            # allow retrigger under sustain 
            if active and use_pedals_for_timing and s64_on == 1 and is_on and (t - start) >= refractory_fr:
                # close previous note at t, then start new one immediately
                s_sec = start / frame_rate
                e_sec = max(t / frame_rate, s_sec + MIN_DUR_SEC)
                inst.notes.append(pretty_midi.Note(
                    velocity=int(start_vel), pitch=21 + p, start=s_sec, end=e_sec
                ))
                # new note
                start = t
                if vel is not None:
                    t0, t1 = t, min(t + 5, T)
                    peak = t0 + int(np.argmax(on[p, t0:t1]))
                    start_vel = int(np.clip(vel[p, peak] * 127.0, 1, 127))
                last_end_t[p] = t  # prevents immediate retrigger spam

            # END
            if active:
                # frame-based drop with FR_OFF
                drop = (t + sustain_frames <= T) and (fr_s[p, t:t+sustain_frames] <= FR_OFF).all()
                want_end = (is_off or drop)

                if use_pedals_for_timing and s64 is not None:
                    if s64_on == 1:
                        # pedal down holds the note
                        want_end = False
                    elif s64_up:
                        # pedal just lifted: end if key is not down
                        want_end = fr_off_now or drop or want_end

                if want_end:
                    s_sec = start / frame_rate
                    e_sec = max(t / frame_rate, s_sec + MIN_DUR_SEC)
                    inst.notes.append(pretty_midi.Note(
                        velocity=int(start_vel), pitch=21 + p, start=s_sec, end=e_sec
                    ))
                    active = False
                    last_end_t[p] = t

            t += 1

        # close tail
        if active:
            s_sec = start / frame_rate
            e_sec = T / frame_rate
            inst.notes.append(pretty_midi.Note(
                velocity=int(start_vel), pitch=21 + p, start=s_sec, end=e_sec
            ))

    pm.instruments.append(inst)
    pm.write(out_path)

def run_one(npz_path, ckpt_path, out_midi):
    model = load_model(ckpt_path)
    on, fr, off, vel, p64, p67 = infer_full_track(npz_path, model)
    print(f"max probs - on:{on.max():.3f} fr:{fr.max():.3f} off:{off.max():.3f} vel:{vel.max():.3f} "
          f"p64:{p64.max():.3f} p67:{p67.max():.3f}")
    os.makedirs(os.path.dirname(out_midi), exist_ok=True)
    decode_to_midi(on, fr, off, out_midi, vel=vel, ped64=p64, ped67=p67)
    print("Wrote:", out_midi)

def run_folder(npz_dir, ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    model = load_model(ckpt_path)
    print("Loading model!", flush=True)
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_midi = os.path.join(out_dir, base + ".mid")
        on, fr, off, vel, p64, p67 = infer_full_track(f, model)
        decode_to_midi(on, fr, off, out_midi, vel=vel, ped64=p64, ped67=p67)
        print("Wrote:", out_midi)

if __name__ == "__main__":
    CKPT    = r"C:...\amt_project\training\checkpoints_velped\best_velped.pth"
    NPZ_DIR = r"C:...\amt_project\preprocessed_full_v3\test"
    OUT_DIR = r"C:...\amt_project\inference\outs_velped_test"
    run_folder(NPZ_DIR, CKPT, OUT_DIR)