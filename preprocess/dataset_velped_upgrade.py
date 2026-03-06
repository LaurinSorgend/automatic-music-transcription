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
        ped64 = data["ped64"] if "ped64" in data.files else np.zeros((1, onset.shape[1]), dtype=np.float32)
        ped66 = data["ped66"] if "ped66" in data.files else np.zeros((1, onset.shape[1]), dtype=np.float32)
        ped67 = data["ped67"] if "ped67" in data.files else np.zeros((1, onset.shape[1]), dtype=np.float32)
        
        T_full = log_mel.shape[1]
        T_seg  = self.segment_frames

        if T_full < T_seg:
            pad = T_seg - T_full
            log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode = "constant")
            onset = np.pad(onset, ((0,0),(0,pad)), mode = "constant")
            frame = np.pad(frame, ((0,0),(0,pad)), mode = "constant")
            offset = np.pad(offset, ((0,0),(0,pad)), mode = "constant")
            vel_on = np.pad(vel_on, ((0,0),(0,pad)), mode = "constant") 
            ped64 = np.pad(ped64, ((0,0),(0,pad)), mode = "constant")
            ped66 = np.pad(ped66, ((0,0),(0,pad)), mode = "constant")
            ped67 = np.pad(ped67, ((0,0),(0,pad)), mode = "constant")
            start = 0
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
        ped64 = ped64[:, start:end]
        ped66 = ped66[:, start:end]
        ped67 = ped67[:, start:end]
        
        return {
            "log_mel":torch.from_numpy(log_mel.astype(np.float32)),
            "onset":torch.from_numpy(onset.astype(np.float32)),
            "frame":torch.from_numpy(frame.astype(np.float32)),
            "offset":torch.from_numpy(offset.astype(np.float32)),
            "vel_on":torch.from_numpy(vel_on.astype(np.float32)),
            "ped64":torch.from_numpy(ped64.astype(np.float32)),
            "ped66":torch.from_numpy(ped66.astype(np.float32)),
            "ped67":torch.from_numpy(ped67.astype(np.float32)),
        }