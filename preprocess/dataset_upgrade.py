# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
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
        log_mel = data["log_mel"]   
        onset = data["onset"]   
        frame = data["frame"]
        offset = data["offset"]
        vel_on = data["vel_on"] if "vel_on" in data.files else np.zeros_like(onset, dtype=np.float32)
        art_on = data["art_on"] if "art_on" in data.files else np.zeros_like(onset, dtype=np.float32)

        T_full = log_mel.shape[1]
        T_seg  = self.segment_frames

        if T_full < T_seg:
            pad = T_seg - T_full
            log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode="constant")
            onset = np.pad(onset, ((0,0),(0,pad)), mode="constant")
            frame = np.pad(frame, ((0,0),(0,pad)), mode="constant")
            offset = np.pad(offset, ((0,0),(0,pad)), mode="constant")
            vel_on = np.pad(vel_on, ((0,0),(0,pad)), mode="constant") 
            art_on = np.pad(art_on, ((0,0),(0,pad)), mode="constant")            
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
        onset = onset[:, start:end]
        frame = frame[:, start:end]
        offset = offset[:, start:end]
        vel_on = vel_on[:, start:end] 
        art_on = art_on[:, start:end] 
        
        return {
            "log_mel":torch.from_numpy(log_mel).float(),
            "onset":torch.from_numpy(onset).float(),
            "frame":torch.from_numpy(frame).float(),
            "offset":torch.from_numpy(offset).float(),
            "vel_on":torch.from_numpy(vel_on.astype(np.float32)),
            "art_on":torch.from_numpy(art_on.astype(np.float32)),
        }