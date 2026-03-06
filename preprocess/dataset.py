# -*- coding: utf-8 -*-
import os, glob, numpy as np, torch
from torch.utils.data import Dataset

class MaestroCropDataset(Dataset):
    def __init__(self, npz_folder, segment_frames=800, mode="train"):
        self.files = sorted(glob.glob(os.path.join(npz_folder, "*.npz")))
        self.segment_frames = segment_frames
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], mmap_mode="r")
        # (80, T_full)
        log_mel = data["log_mel"]   
        onset = data["onset"]
        frame = data["frame"]
        offset = data["offset"]

        T_full = log_mel.shape[1]
        T_seg  = self.segment_frames

        if T_full < T_seg:
            # Padding for if clip is shorter
            pad = T_seg - T_full
            log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode="constant")
            onset = np.pad(onset, ((0,0),(0,pad)), mode="constant")
            frame = np.pad(frame, ((0,0),(0,pad)), mode="constant")
            offset = np.pad(offset, ((0,0),(0,pad)), mode="constant")
            start = 0
            end = T_seg
        else:
            if self.mode == "train":
                # Random crop
                start = np.random.randint(0, T_full - T_seg + 1)
            else:
                start = (T_full - T_seg) // 2

        end = start + T_seg
        # Slice out the segment
        log_mel = log_mel[:, start:end]
        onset = onset  [:, start:end]
        frame = frame  [:, start:end]
        offset = offset [:, start:end]

        return {
            "log_mel":torch.from_numpy(log_mel).float(),
            "onset":torch.from_numpy(onset).float(),
            "frame":torch.from_numpy(frame).float(),
            "offset":torch.from_numpy(offset).float(),
        }