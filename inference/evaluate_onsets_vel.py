# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import pretty_midi

ONSET_TOL = 0.050   
OFF_ABS   = 0.050
OFF_REL   = 0.20     

def parse_window_from_name(name):
    base = os.path.splitext(os.path.basename(name))[0]
    if base.endswith("_0-8"):
        return 0.0, 8.0, base[:-4]  
    return None, None, base

def midi_to_notes(pm):
    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity))
    notes.sort(key=lambda x: (x[2], x[0]))
    return notes

def clip_ref_notes(ref_notes, t0, t1):
    out = []
    for (s, e, p, v) in ref_notes:
        if e <= t0 or s >= t1:
            continue
        s2 = max(s, t0)
        e2 = min(e, t1)
        out.append((s2, e2, p, v))
    return out

def match_notes(pred, ref, onset_only=True):
    TP = 0; used = set()
    ref_by_pitch = {}
    for i, n in enumerate(ref):
        ref_by_pitch.setdefault(n[2], []).append((i, n))
    for (po, pf, pp, _) in pred:
        best = (-1, 1e9)  # (idx, onset_error)
        for (idx, (ro, rf, rp, _)) in ref_by_pitch.get(pp, []):
            if idx in used: continue
            if abs(po - ro) > ONSET_TOL: continue
            if not onset_only:
                ref_dur = max(rf - ro, 1e-6)
                off_tol = max(OFF_ABS, OFF_REL * ref_dur)
                if abs(pf - rf) > off_tol: continue
            dt = abs(po - ro)
            if dt < best[1]: best = (idx, dt)
        if best[0] >= 0:
            used.add(best[0]); TP += 1
    FP = len(pred) - TP
    FN = len(ref) - TP
    P = TP/(TP+FP) if TP+FP>0 else 0.0
    R = TP/(TP+FN) if TP+FN>0 else 0.0
    F = 2*P*R/(P+R) if P+R>0 else 0.0
    return P, R, F

def match_notes_with_pairs(pred, ref, onset_only=True):
    used = set()
    pairs = []
    ref_by_pitch = {}
    for i, n in enumerate(ref):
        ref_by_pitch.setdefault(n[2], []).append((i, n))

    for pi, (po, pf, pp, _) in enumerate(pred):
        best = (-1, 1e9)  # (ri, onset_error)
        for (ri, (ro, rf, rp, _)) in ref_by_pitch.get(pp, []):
            if ri in used:
                continue
            if abs(po - ro) > ONSET_TOL:
                continue
            if not onset_only:
                ref_dur = max(rf - ro, 1e-6)
                off_tol = max(OFF_ABS, OFF_REL * ref_dur)
                if abs(pf - rf) > off_tol:
                    continue
            dt = abs(po - ro)
            if dt < best[1]:
                best = (ri, dt)
        if best[0] >= 0:
            used.add(best[0])
            pairs.append((pi, best[0]))
    return pairs

def evaluate_folder(pred_dir, data_root, csv_path, onset_only=True):
    df = pd.read_csv(csv_path)
    # map base audio name -> MIDI path
    m = {}
    for _, r in df.iterrows():
        base = os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        midi_path = os.path.normpath(os.path.join(data_root, r["midi_filename"]))
        m[base] = midi_path

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.mid")))
    Ps=Rs=Fs=0.0; N=0; missing=0

    for pf in preds:
        t0, t1, base = parse_window_from_name(pf)
        ref_path = m.get(base)
        if not ref_path or not os.path.exists(ref_path):
            missing += 1; continue

        pred_notes = midi_to_notes(pretty_midi.PrettyMIDI(pf))
        ref_all = midi_to_notes(pretty_midi.PrettyMIDI(ref_path))
        if t0 is not None:
            ref_notes = clip_ref_notes(ref_all, t0, t1)
        else:
            ref_notes = ref_all

        P, R, F = match_notes(pred_notes, ref_notes, onset_only=onset_only)
        Ps += P; Rs += R; Fs += F; N += 1

    if N == 0:
        print("No files evaluated.")
        return 0,0,0
    print(f"Files evaluated: {N} (missing refs: {missing})")
    print(f"Note {'Onset' if onset_only else 'Onset+Offset'} F1: "
          f"P {Ps/N:.3f} / R {Rs/N:.3f} / F1 {Fs/N:.3f}")
    return Ps/N, Rs/N, Fs/N

def evaluate_folder_velocityF1(pred_dir, data_root, csv_path,
                               onset_only=True, vel_tol=0.10, calibrate=True):
    df = pd.read_csv(csv_path)

    # audio base -> reference MIDI path
    ref_map = {}
    for _, r in df.iterrows():
        base = os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        ref_map[base] = os.path.normpath(os.path.join(data_root, r["midi_filename"]))

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.mid")))
    Ps = Rs = Fs = 0.0
    N = 0
    missing = 0

    for pf in preds:
        t0, t1, base = parse_window_from_name(pf)
        ref_path = ref_map.get(base)
        if not ref_path or not os.path.exists(ref_path):
            missing += 1
            continue

        pred_notes = midi_to_notes(pretty_midi.PrettyMIDI(pf))
        ref_all    = midi_to_notes(pretty_midi.PrettyMIDI(ref_path))
        ref_notes  = clip_ref_notes(ref_all, t0, t1) if t0 is not None else ref_all

        # pairs matched on timing only (no velocity constraint yet)
        pairs = match_notes_with_pairs(pred_notes, ref_notes, onset_only=onset_only)

        # handle degenerate cases per file
        n_pred = len(pred_notes)
        n_ref  = len(ref_notes)
        if n_pred == 0 and n_ref == 0:
            # define as perfect match for this file
            Ps += 1.0; Rs += 1.0; Fs += 1.0; N += 1
            continue
        if n_pred == 0 or n_ref == 0 or len(pairs) == 0:
            # no usable matches: P=0 if no preds, R=0 if no refs (P/R/F -> 0)
            N += 1
            continue

        # xollect velocities for matched pairs, normalize to [0,1]
        pv = np.array([pred_notes[pi][3] for (pi, _) in pairs], dtype=np.float32) / 127.0
        rv = np.array([ref_notes [ri][3] for (_,  ri) in pairs], dtype=np.float32) / 127.0

        # Optional per-piece affine calibration v' = m*v + b (least squares)
        if calibrate and len(pv) >= 2:
            X = np.stack([pv, np.ones_like(pv)], axis=1)   # [pv, 1]
            m, b = np.linalg.lstsq(X, rv, rcond=None)[0]
            pv_hat = np.clip(m * pv + b, 0.0, 1.0)
        else:
            pv_hat = pv  

        # Keep only pairs whose velocities are close within tolerance
        good = np.abs(pv_hat - rv) <= vel_tol
        TP = int(good.sum())

        # Per-file P/R/F 
        P = TP / n_pred if n_pred > 0 else 0.0
        R = TP / n_ref  if n_ref  > 0 else 0.0
        F = (2*P*R/(P+R)) if (P+R) > 0 else 0.0

        Ps += P; Rs += R; Fs += F; N += 1

    if N == 0:
        print("No files evaluated.")
        return 0.0, 0.0, 0.0

    label = "Onset+Velocity" if onset_only else "Onset+Offset+Velocity"
    print(f"Files evaluated: {N} (missing refs: {missing})")
    print(f"Note {label} F1: P {Ps/N:.3f} / R {Rs/N:.3f} / F1 {Fs/N:.3f} "
          f"(vel_tol={vel_tol:.2f}, calibrate={calibrate})")
    return Ps/N, Rs/N, Fs/N

def save_summary(pred_dir, data_root, csv_path, ckpt_path, th_on, th_fr, th_off, out_txt):
    import datetime, json, os
    # evaluate both variants
    P1,R1,F1 = evaluate_folder(pred_dir, data_root, csv_path, onset_only=True)
    P2,R2,F2 = evaluate_folder(pred_dir, data_root, csv_path, onset_only=False)
    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "pred_dir": pred_dir,
        "ckpt": ckpt_path,
        "thresholds": {"TH_ON": th_on, "TH_FR": th_fr, "TH_OFF": th_off},
        "onset_only": {"P": P1, "R": R1, "F1": F1},
        "onset_offset": {"P": P2, "R": R2, "F1": F2}
    }
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", out_txt)
    return summary

if __name__ == "__main__":
    DATA_ROOT = r"C:...\amt_project\data\MAESTRO-V3.0.0"
    CSV       = rf"{DATA_ROOT}\maestro-v3.0.0.csv"
    PRED_DIR  = r"C:...\amt_project\inference\outs_oc_vel_test"
    evaluate_folder(PRED_DIR, DATA_ROOT, CSV, onset_only=True)   # onset-only
    evaluate_folder(PRED_DIR, DATA_ROOT, CSV, onset_only=False)  # onset+offset
    evaluate_folder_velocityF1(PRED_DIR, DATA_ROOT, CSV, onset_only=True,  vel_tol=0.10, calibrate=True)
    evaluate_folder_velocityF1(PRED_DIR, DATA_ROOT, CSV, onset_only=False, vel_tol=0.10, calibrate=True)