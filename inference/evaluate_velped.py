# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import pretty_midi

ONSET_TOL = 0.050    
OFF_ABS = 0.050
OFF_REL = 0.20     

FRAME_RATE = 100

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


def midi_end_time(pm):
    end = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            end = max(end, n.end)
        for c in inst.control_changes:
            end = max(end, c.time)
    return end

def cc_timeline(pm: pretty_midi.PrettyMIDI, cc_num: int,
                fs: int = FRAME_RATE, thr: int = 64,
                t0: float | None = None, t1: float | None = None) -> np.ndarray:
    # collect all events for this CC
    events = []
    for inst in pm.instruments:
        for c in inst.control_changes:
            if c.number == cc_num:
                events.append((c.time, c.value))
    # If nothing, return zeros
    if not events and t0 is None and t1 is None:
        T = max(1, int(round(midi_end_time(pm) * fs)))
        return np.zeros(T, np.int8)
    if not events:
        T = max(1, int(round((t1 - t0) * fs))) if (t0 is not None and t1 is not None) else 1
        return np.zeros(T, np.int8)

    events.sort(key=lambda x: x[0])
    
    # decide window
    if t0 is None:
        t0 = 0.0
    if t1 is None:
        t1 = midi_end_time(pm)
    T = max(1, int(round((t1 - t0) * fs)))

    # step through events, writing state into timeline
    y = np.zeros(T, np.int8)
    i_prev = 0
    # initial state from first event 
    state = 1 if events[0][1] >= thr else 0
    # fill from start until first in-window event
    for time, val in events:
        if time < t0:
            state = 1 if val >= thr else 0
            continue
        i = int(round((time - t0) * fs))
        i = max(0, min(T, i))
        if i > i_prev:
            y[i_prev:i] = state
            i_prev = i
        state = 1 if val >= thr else 0
        if i >= T:
            break
    # tail
    if i_prev < T:
        y[i_prev:T] = state
    return y

def evaluate_folder(pred_dir, data_root, csv_path, onset_only=True):
    df = pd.read_csv(csv_path)
    base2ref = {}
    for _, r in df.iterrows():
        base = os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        base2ref[base] = os.path.normpath(os.path.join(data_root, r["midi_filename"]))

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.mid")))
    Ps=Rs=Fs=0.0; N=0; missing=0

    for pf in preds:
        t0, t1, base = parse_window_from_name(pf)
        ref_path = base2ref.get(base)
        if not ref_path or not os.path.exists(ref_path):
            missing += 1; continue

        pred_notes = midi_to_notes(pretty_midi.PrettyMIDI(pf))
        ref_all    = midi_to_notes(pretty_midi.PrettyMIDI(ref_path))
        ref_notes  = clip_ref_notes(ref_all, t0, t1) if t0 is not None else ref_all

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
            # no usable matches: P=0 if no pred, R=0 if no refs (P/R/F is 0)
            N += 1
            continue

        # Collect velocities for matched pairs, normalize to [0,1]
        pv = np.array([pred_notes[pi][3] for (pi, _) in pairs], dtype=np.float32) / 127.0
        rv = np.array([ref_notes [ri][3] for (_,  ri) in pairs], dtype=np.float32) / 127.0

        # Optional per-piece affine calibration v' = m*v + b (least squares)
        if calibrate and len(pv) >= 2:
            X = np.stack([pv, np.ones_like(pv)], axis=1)   # [pv, 1]
            m, b = np.linalg.lstsq(X, rv, rcond=None)[0]
            pv_hat = np.clip(m * pv + b, 0.0, 1.0)
        else:
            pv_hat = pv  # no calibration

        # Keep only pairs whose velocities are close within tolerance
        good = np.abs(pv_hat - rv) <= vel_tol
        TP = int(good.sum())

        # Per-file P/R/F (same style as your existing evaluate_folder)
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

# pedal (event-based)
def _cc_events(pm: pretty_midi.PrettyMIDI, cc_num: int):
    ev = []
    for inst in pm.instruments:
        for c in inst.control_changes:
            if c.number == cc_num:
                ev.append((c.time, c.value))
    ev.sort(key=lambda x: x[0])
    return ev

def _cc_value_timeline(pm: pretty_midi.PrettyMIDI, cc_num: int, fs: int, t0: float, t1: float):
    events = _cc_events(pm, cc_num)
    T = max(1, int(round((t1 - t0) * fs)))
    y = np.zeros(T, dtype=np.int16)
    if not events:
        return y
    i = 0
    cur = 0
    # prime with last event at/before t0, else first event's value
    for (te, ve) in events:
        if te <= t0: cur = ve
        else: break
    for t in range(T):
        now = t0 + t / fs
        while i < len(events) and events[i][0] <= now:
            cur = events[i][1]
            i += 1
        y[t] = cur
    return y  # 0..127

def _hysteresis_binarize(values_0_127: np.ndarray, thr_on=64, thr_off=60):
    out = np.zeros_like(values_0_127, dtype=np.uint8)
    on = False
    for t, v in enumerate(values_0_127):
        if not on and v >= thr_on: on = True
        elif on and v <= thr_off:   on = False
        out[t] = 1 if on else 0
    return out

def _debounce_merge(state01: np.ndarray, fs: int, min_on_ms=80, min_off_ms=60):
    hop_ms = 1000.0 / fs
    def runs(arr):
        idx = np.flatnonzero(np.diff(np.concatenate(([0], arr, [0]))))
        starts, ends = idx[::2], idx[1::2]
        return list(zip(starts, ends))
    on_runs = [(a,b) for (a,b) in runs(state01) if (b-a)*hop_ms >= min_on_ms]
    merged = []
    i = 0
    while i < len(on_runs):
        a, b = on_runs[i]
        while i+1 < len(on_runs) and (on_runs[i+1][0] - b)*hop_ms < min_off_ms:
            b = on_runs[i+1][1]
            i += 1
        merged.append((a,b)); i += 1
    out = np.zeros_like(state01, dtype=np.uint8)
    for a,b in merged: out[a:b] = 1
    return out

def _state_to_events(state01: np.ndarray, fs: int):
    downs = np.where((state01[1:] == 1) & (state01[:-1] == 0))[0] + 1
    ups   = np.where((state01[1:] == 0) & (state01[:-1] == 1))[0] + 1
    return downs / fs, ups / fs

def _match_events(pred_times, ref_times, tol=0.050):
    pred = np.asarray(pred_times, dtype=np.float32)
    ref  = np.asarray(ref_times,  dtype=np.float32)
    if len(pred)==0 and len(ref)==0: return 1.0, 1.0, 1.0
    if len(pred)==0: return 0.0, 0.0, 0.0
    if len(ref)==0:  return 0.0, 0.0, 0.0
    used = np.zeros(len(ref), dtype=bool)
    TP = 0
    for p in pred:
        i = np.argmin(np.abs(ref - p))
        if not used[i] and abs(ref[i] - p) <= tol:
            used[i] = True
            TP += 1
    FP = len(pred) - TP
    FN = len(ref)  - TP
    P = TP/(TP+FP) if (TP+FP)>0 else 0.0
    R = TP/(TP+FN) if (TP+FN)>0 else 0.0
    F = (2*P*R/(P+R)) if (P+R)>0 else 0.0
    return P, R, F

def evaluate_pedals_folder(pred_dir, data_root, csv_path, fs=FRAME_RATE, thr=64, lanes=(64, 67)):
    df = pd.read_csv(csv_path)
    base2ref = {}
    for _, r in df.iterrows():
        base = os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        base2ref[base] = os.path.normpath(os.path.join(data_root, r["midi_filename"]))

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.mid")))
    sums = {k: {"P":0.0,"R":0.0,"F":0.0} for k in lanes}
    N = 0; missing = 0

    for pf in preds:
        t0, t1, base = parse_window_from_name(pf)
        ref_path = base2ref.get(base)
        if not ref_path or not os.path.exists(ref_path):
            missing += 1; continue

        pred_pm = pretty_midi.PrettyMIDI(pf)
        ref_pm  = pretty_midi.PrettyMIDI(ref_path)

        # If no window provided, compare over the union duration
        if t0 is None or t1 is None:
            t0 = 0.0
            t1 = max(midi_end_time(pred_pm), midi_end_time(ref_pm))

        for cc in lanes:
            y_pred = cc_timeline(pred_pm, cc, fs=fs, thr=thr, t0=t0, t1=t1)
            y_ref  = cc_timeline(ref_pm,  cc, fs=fs, thr=thr, t0=t0, t1=t1)

            # align length (just in case of rounding differences)
            T = min(len(y_pred), len(y_ref))
            y_pred = y_pred[:T]; y_ref = y_ref[:T]

            TP = int(np.logical_and(y_pred==1, y_ref==1).sum())
            FP = int(np.logical_and(y_pred==1, y_ref==0).sum())
            FN = int(np.logical_and(y_pred==0, y_ref==1).sum())

            P = TP/(TP+FP) if (TP+FP)>0 else 0.0
            R = TP/(TP+FN) if (TP+FN)>0 else 0.0
            F = (2*P*R/(P+R)) if (P+R)>0 else 0.0

            sums[cc]["P"] += P
            sums[cc]["R"] += R
            sums[cc]["F"] += F

        N += 1

    if N == 0:
        print("No files evaluated for pedals.")
        return {64:(0,0,0), 66:(0,0,0), 67:(0,0,0)}

    results = {}
    print(f"Files evaluated (pedals): {N} (missing refs: {missing})")
    for cc in lanes:
        P = sums[cc]["P"]/N; R = sums[cc]["R"]/N; F = sums[cc]["F"]/N
        results[cc] = (P,R,F)
        name = {64:"CC64 sustain", 67:"CC67 una corda"}[cc]
        print(f"{name} F1: P {P:.3f} / R {R:.3f} / F1 {F:.3f}")
    return results

def evaluate_pedals_events_folder(pred_dir, data_root, csv_path,
                                  fs=FRAME_RATE, tol=0.050,
                                  thr_on=64, thr_off=60,
                                  min_on_ms=80, min_off_ms=60,
                                  lanes=(64, 67)):
    df = pd.read_csv(csv_path)
    base2ref = {}
    for _, r in df.iterrows():
        base = os.path.splitext(os.path.basename(r["audio_filename"]))[0]
        base2ref[base] = os.path.normpath(os.path.join(data_root, r["midi_filename"]))

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.mid")))
    sums = {k: {"down": {"P":0.0,"R":0.0,"F":0.0},
                "up":   {"P":0.0,"R":0.0,"F":0.0}} for k in lanes}
    N = 0; missing = 0

    for pf in preds:
        t0, t1, base = parse_window_from_name(pf)
        ref_path = base2ref.get(base)
        if not ref_path or not os.path.exists(ref_path):
            missing += 1; continue

        pred_pm = pretty_midi.PrettyMIDI(pf)
        ref_pm  = pretty_midi.PrettyMIDI(ref_path)

        # window: if none, compare over union duration
        if t0 is None or t1 is None:
            t0 = 0.0
            t1 = max(midi_end_time(pred_pm), midi_end_time(ref_pm))

        for cc in lanes:
            # timelines of raw 0..127 values
            v_pred = _cc_value_timeline(pred_pm, cc, fs, t0, t1)
            v_ref  = _cc_value_timeline(ref_pm,  cc, fs, t0, t1)

            # hysteresis -> binary; then debounce/merge
            b_pred = _debounce_merge(_hysteresis_binarize(v_pred, thr_on, thr_off), fs, min_on_ms, min_off_ms)
            b_ref  = _debounce_merge(_hysteresis_binarize(v_ref,  thr_on, thr_off), fs, min_on_ms, min_off_ms)

            # events
            pd_down, pd_up = _state_to_events(b_pred, fs)
            rf_down, rf_up = _state_to_events(b_ref,  fs)

            # score
            P,R,F = _match_events(pd_down, rf_down, tol); sums[cc]["down"]["P"] += P; sums[cc]["down"]["R"] += R; sums[cc]["down"]["F"] += F
            P,R,F = _match_events(pd_up,   rf_up,   tol); sums[cc]["up"]["P"]   += P; sums[cc]["up"]["R"]   += R; sums[cc]["up"]["F"]   += F

        N += 1

    if N == 0:
        print("No files evaluated for pedals (events).")
        return {k: {"down":(0,0,0), "up":(0,0,0)} for k in lanes}

    results = {}
    print(f"Files evaluated (pedal events): {N} (missing refs: {missing}); tol={tol*1000:.0f} ms; thr_on={thr_on}, thr_off={thr_off}; "
          f"debounce ON≥{min_on_ms}ms, OFFgap<{min_off_ms}ms merged.")
    for cc in lanes:
        name = {64:"CC64 sustain", 67:"CC67 una corda"}.get(cc, f"CC{cc}")
        Pd = sums[cc]["down"]["P"]/N; Rd = sums[cc]["down"]["R"]/N; Fd = sums[cc]["down"]["F"]/N
        Pu = sums[cc]["up"]["P"]/N;   Ru = sums[cc]["up"]["R"]/N;   Fu = sums[cc]["up"]["F"]/N
        results[cc] = {"down": (Pd,Rd,Fd), "up": (Pu,Ru,Fu)}
        print(f"{name} Down  F1: P {Pd:.3f} / R {Rd:.3f} / F1 {Fd:.3f}")
        print(f"{name} Up    F1: P {Pu:.3f} / R {Ru:.3f} / F1 {Fu:.3f}")
    return results

if __name__ == "__main__":
    DATA_ROOT = r"C:...\amt_project\data\MAESTRO-V3.0.0"
    CSV       = rf"{DATA_ROOT}\maestro-v3.0.0.csv"
    PRED_DIR  = r"C:...\amt_project\inference\outs_velped_test"
    evaluate_folder(PRED_DIR, DATA_ROOT, CSV, onset_only=True)   # onset-only
    evaluate_folder(PRED_DIR, DATA_ROOT, CSV, onset_only=False)  # onset+offset
    # Velocity-aware F1 
    evaluate_folder_velocityF1(PRED_DIR, DATA_ROOT, CSV, onset_only=True,  vel_tol=0.10, calibrate=True)
    evaluate_folder_velocityF1(PRED_DIR, DATA_ROOT, CSV, onset_only=False, vel_tol=0.10, calibrate=True)
    # Pedals — event-based 
    evaluate_pedals_events_folder(PRED_DIR, DATA_ROOT, CSV,
                                  fs=FRAME_RATE, tol=0.050,
                                  thr_on=64, thr_off=60,
                                  min_on_ms=80, min_off_ms=60,
                                  lanes=(64, 67))