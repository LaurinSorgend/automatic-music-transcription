# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
import pretty_midi

from model.onset_conformer_amt import AMTConformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_MELS      = 128
FRAME_RATE  = 100     # 10ms hop
SEG_FRAMES  = 800     # 8s windows
HOP_FRAMES  = 200     # 2s hop

TH_ON  = 0.14
TH_FR  = 0.32
TH_OFF = 0.26
MIN_DUR_SEC = 0.06    # drop short notes

def load_model(ckpt_path: str) -> AMTConformer:
    model = AMTConformer(n_mels=N_MELS).to(DEVICE).eval()
    ck = torch.load(ckpt_path, map_location=DEVICE)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    return model

@torch.no_grad()
def infer_full_track(npz_path: str, model: AMTConformer):
    d = np.load(npz_path, mmap_mode="r")
    mel = d["log_mel"]            # (128, T_full)
    T_full = mel.shape[1]

    # stitchers over time (T_full, 88) then transpose at end
    on_sum  = np.zeros((T_full, 88), dtype=np.float32)
    fr_sum  = np.zeros((T_full, 88), dtype=np.float32)
    off_sum = np.zeros((T_full, 88), dtype=np.float32)
    vel_sum = np.zeros((T_full, 88), dtype=np.float32)
    cnt     = np.zeros((T_full, 1),  dtype=np.float32)

    t0 = 0
    while t0 < T_full:
        t1 = min(t0 + SEG_FRAMES, T_full)
        seg = mel[:, t0:t1]
        valid = seg.shape[1]
        if valid < SEG_FRAMES:
            seg = np.pad(seg, ((0,0),(0, SEG_FRAMES - valid)), mode="constant")

        x = torch.from_numpy(seg).unsqueeze(0).float().to(DEVICE) 
        on_logit, fr_logit, off_logit, vel_logit, _  = model(x)                  
        on = torch.sigmoid(on_logit).squeeze(0).cpu().numpy()     
        fr = torch.sigmoid(fr_logit).squeeze(0).cpu().numpy()
        off = torch.sigmoid(off_logit).squeeze(0).cpu().numpy()
        vel = torch.sigmoid(vel_logit).squeeze(0).cpu().numpy()

        on_sum[t0:t1] += on[:valid]
        fr_sum[t0:t1] += fr[:valid]
        off_sum[t0:t1] += off[:valid]
        vel_sum[t0:t1] += vel[:valid]
        cnt[t0:t1] += 1.0

        t0 += HOP_FRAMES

    on_prob = (on_sum  / np.clip(cnt, 1e-6, None)).T  
    fr_prob = (fr_sum  / np.clip(cnt, 1e-6, None)).T
    off_prob = (off_sum / np.clip(cnt, 1e-6, None)).T
    vel_prob = (vel_sum / np.clip(cnt, 1e-6, None)).T
    return on_prob, fr_prob, off_prob, vel_prob

def decode_to_midi(on, fr, off, out_path, frame_rate=FRAME_RATE, default_vel=64, vel=None):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand

    def smooth_time(prob, width=3):
        # prob: (88, T)
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

    on_s = smooth_time(on, 3)
    fr_s = smooth_time(fr, 3)
    off_s= smooth_time(off, 3)

    # decoding params
    sustain_frames  = 3 
    refractory_fr   = 5   

    last_end_t = np.full(88, -9999, dtype=int)

    for p in range(88):
        active = False
        start = 0
        T = on.shape[1]

        t = 0
        while t < T:
            is_on = on_s[p, t]  > TH_ON
            is_fr = fr_s[p, t]  > TH_FR
            is_off = off_s[p, t] > TH_OFF

            if (not active) and is_on and is_fr and (t - last_end_t[p] >= refractory_fr):
                # tiny sustain: next few frames should remain framed
                if t + sustain_frames <= T and (fr_s[p, t+1:t+sustain_frames] > TH_FR).all():
                    # optional local onset peak to avoid double triggers
                    if t == 0 or on_s[p, t] >= on_s[p, t-1]:
                        active = True
                        start = t

            if active:
                drop = (t + sustain_frames <= T) and (fr_s[p, t:t+sustain_frames] <= TH_FR).all()
                if is_off or drop:
                    s = start / frame_rate
                    e = max(t / frame_rate, s + MIN_DUR_SEC)

                    # choose velocity from local onset peak (first 50 ms window)
                    if vel is not None:
                        t0 = start
                        t1 = min(start + 5, T)  # 50 ms @ 100 Hz
                        peak = t0 + int(np.argmax(on_s[p, t0:t1]))
                        v = int(np.clip((vel[p, peak] if vel is not None else default_vel/127.0) * 127.0, 1, 127))
                    else:
                        v = int(np.clip(default_vel, 1, 127))

                    inst.notes.append(pretty_midi.Note(velocity=v, pitch=21 + p, start=s, end=e))
                    active = False
                    last_end_t[p] = t

            t += 1

        # close if still active at the end
        if active:
            s = start / frame_rate
            e = T / frame_rate
            if vel is not None:
                t0 = start
                t1 = min(start + 5, T)
                peak = t0 + int(np.argmax(on_s[p, t0:t1]))
                v = int(np.clip(vel[p, peak] * 127.0, 1, 127))
            else:
                v = int(np.clip(default_vel, 1, 127))
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=21 + p, start=s, end=e))

    pm.instruments.append(inst)
    pm.write(out_path)

def run_one(npz_path, ckpt_path, out_midi):
    model = load_model(ckpt_path)
    on, fr, off, vel = infer_full_track(npz_path, model)
    print(f"max probs - on:{on.max():.3f} fr:{fr.max():.3f} off:{off.max():.3f} vel:{vel.max():.3f}")
    print("onset frames above thr:", int((on > TH_ON).sum()))
    os.makedirs(os.path.dirname(out_midi), exist_ok=True)
    decode_to_midi(on, fr, off, out_midi, vel=vel)
    print("Wrote:", out_midi)

def run_folder(npz_dir, ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    model = load_model(ckpt_path)
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_midi = os.path.join(out_dir, base + ".mid")
        on, fr, off, vel = infer_full_track(f, model)
        decode_to_midi(on, fr, off, out_midi, vel=vel)
        print("Wrote:", out_midi)

if __name__ == "__main__":
    CKPT    = r"C:...\amt_project\training\checkpoints_new\best_onsets.pth"
    NPZ_DIR = r"C:...\amt_project\preprocessed_full_v2\test"
    OUT_DIR = r"C:...\amt_project\inference\outs_oc_vel_test"
    run_folder(NPZ_DIR, CKPT, OUT_DIR)