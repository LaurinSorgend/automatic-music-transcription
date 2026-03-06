# -*- coding: utf-8 -*-
print("velped_train.py is running; __name__ =", __name__)

import os
import torch
from torch import amp
from torch.utils.data import DataLoader
from preprocess.dataset_velped_upgrade import MaestroCropDataset
from model.velped_conformer_amt import AMTConformer 

# paths
TRAIN_DIR = r"C:...\amt_project\preprocessed_full_v3\train"
VAL_DIR   = r"C:...\amt_project\preprocessed_full_v3\val"
TEST_DIR  = r"C:...\amt_project\preprocessed_full_v3\test"

# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 40
BATCH_SIZE = 8
SEGMENT_FRAMES = 800      
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# datasets
train_ds = MaestroCropDataset(TRAIN_DIR, segment_frames=SEGMENT_FRAMES, mode="train")
val_ds = MaestroCropDataset(VAL_DIR,   segment_frames=SEGMENT_FRAMES, mode="val")

# loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True, drop_last=False)

# loss
def make_losses():
    # Pos weights 
    pw_on = torch.tensor(5.0, device=DEVICE)
    pw_off = torch.tensor(3.0, device=DEVICE)
    pw_p64 = torch.tensor(1.0, device=DEVICE)  
    pw_p67 = torch.tensor(2.0, device=DEVICE)  

    bce_on = torch.nn.BCEWithLogitsLoss(pos_weight=pw_on)
    bce_fr = torch.nn.BCEWithLogitsLoss()
    bce_off = torch.nn.BCEWithLogitsLoss(pos_weight=pw_off)
    bce_p64 = torch.nn.BCEWithLogitsLoss(pos_weight=pw_p64)
    bce_p67 = torch.nn.BCEWithLogitsLoss(pos_weight=pw_p67)
    return bce_on, bce_fr, bce_off, bce_p64, bce_p67

bce_on, bce_fr, bce_off, bce_p64, bce_p67 = make_losses()

# velocity mask
def masked_l1(pred, target, mask):
    num = (torch.abs(pred - target) * mask).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den

def compute_loss(on_logits, fr_logits, off_logits, vel_logits,
                 ped64_logits, ped67_logits,
                 on_t, fr_t, off_t, vel_t, ped64_t, ped67_t,
                 w_on=1.2, w_fr=0.5, w_off=0.5, w_vel=0.5, w_ped=0.25):
    
    on_logits = on_logits.transpose(1, 2)
    fr_logits = fr_logits.transpose(1, 2)
    off_logits = off_logits.transpose(1, 2)
    ped64_logits = ped64_logits.transpose(1, 2)
    ped67_logits = ped67_logits.transpose(1, 2)
                                      
    l_on = bce_on(on_logits,  on_t)
    l_fr = bce_fr(fr_logits,  fr_t)
    l_off = bce_off(off_logits, off_t)
    
    vel_pred = torch.sigmoid(vel_logits).transpose(1, 2)  
    on_mask  = on_t
    l_vel = masked_l1(vel_pred, vel_t, on_mask)

    # pedals - average of CC64 + CC67
    l_p64 = bce_p64(ped64_logits, ped64_t)  
    l_p67 = bce_p67(ped67_logits, ped67_t)
    l_ped = 0.5 * (l_p64 + l_p67)
    
    loss = w_on*l_on + w_fr*l_fr + w_off*l_off + w_vel*l_vel + w_ped*l_ped
    return loss, {
        "on": l_on.item(),
        "fr": l_fr.item(),
        "off": l_off.item(),
        "vel": l_vel.item(),
        "ped": l_ped.item()
    }

# training
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["log_mel"].to(DEVICE)   
        on_t = batch["onset"].to(DEVICE)    
        fr_t = batch["frame"].to(DEVICE)
        off_t = batch["offset"].to(DEVICE)
        vel_t = batch["vel_on"].to(DEVICE)
        ped64_t = batch["ped64"].to(DEVICE) 
        ped67_t = batch["ped67"].to(DEVICE)        
        
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda'):
            on_l, fr_l, off_l, vel_l, p64_l, p67_l = model(x, teach_on=on_t)

            loss, _ = compute_loss(
                on_logits=on_l, fr_logits=fr_l, off_logits=off_l,
                vel_logits=vel_l, ped64_logits=p64_l, ped67_logits=p67_l,
                on_t=on_t, fr_t=fr_t, off_t=off_t, vel_t=vel_t,
                ped64_t=ped64_t, ped67_t=ped67_t
            )               
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["log_mel"].to(DEVICE)
            on_t = batch["onset"].to(DEVICE)
            fr_t = batch["frame"].to(DEVICE)
            off_t = batch["offset"].to(DEVICE)
            vel_t = batch["vel_on"].to(DEVICE)  
            ped64_t = batch["ped64"].to(DEVICE)
            ped67_t = batch["ped67"].to(DEVICE)           
            
            on_l, fr_l, off_l, vel_l, p64_l, p67_l = model(x)
            loss, _ = compute_loss(
                on_logits=on_l, fr_logits=fr_l, off_logits=off_l,
                vel_logits=vel_l, ped64_logits=p64_l, ped67_logits=p67_l,
                on_t=on_t, fr_t=fr_t, off_t=off_t, vel_t=vel_t,
                ped64_t=ped64_t, ped67_t=ped67_t
            )
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# checkpoints
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__),"checkpoints_velped")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
last_ckpt = os.path.join(CHECKPOINT_DIR, "last_velped.pth")
best_ckpt = os.path.join(CHECKPOINT_DIR, "best_velped.pth")

def main():
    # model + opt + scaler
    model = AMTConformer(n_mels=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler('cuda')  # DEPRECATED: scaler = GradScaler()

    # optional resume
    start_epoch   = 1
    best_val_loss = float("inf")
    if os.path.exists(last_ckpt):
        try:
            ckpt = torch.load(last_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", best_val_loss)
            print(f"Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.4f})")
        except Exception as e:
            print("Resume failed -", e)

    # epoch loop
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss   = validate(model, val_loader)

        print(f"Epoch {epoch:02d} / Train Loss: {train_loss:.4f} / Val Loss: {val_loss:.4f}")

        # save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": val_loss,
        }, last_ckpt)

        # if best so far, overwrite “best.pth”
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_ckpt)
            print(f"  Yippee! New best saved: {best_ckpt}")

if __name__ == "__main__":
    main()