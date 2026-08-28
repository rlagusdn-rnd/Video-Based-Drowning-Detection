"""
VideoMAEv2 K710 backbone + LoRA(qkv) + Linear head fine-tuning.

이전 head-only baseline(head_finetune_best64.pt)과 동일 데이터/split에서
LoRA가 백본 적응에 효과 있는지 검증.

학습 대상: LoRA params (~295K) + head (~1.5K) = ~0.34% of 86M
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from data.video_dataset import VideoDataset
from models.lora_backbone import VideoMAEv2_LoRA_Linear


# ===== Paths =====
DATA_ROOT = "/root/workspace/Drowning-Detection-VLM/datasets"
OUT_CKPT  = "checkpoints/lora_qkv_best.pt"
OUT_CURVE = "checkpoints/lora_loss_curve.png"

# ===== Input spec (extract_videomae_v2.py와 동일 — 학습/feature 일관성) =====
NUM_FRAMES = 16
IMG_SIZE   = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

# ===== Hyperparameters =====
NUM_CLASSES        = 2
BATCH_SIZE         = 2
ACCUMULATION_STEPS = 8         # effective batch = BATCH_SIZE * ACCUMULATION_STEPS = 16
NUM_EPOCHS         = 15
LR_LORA            = 2e-4      # LoRA는 처음부터 학습 → 약간 높은 lr
LR_HEAD            = 1e-4      # head는 best64에서 출발 → 낮은 lr로 미세 조정
WEIGHT_DECAY       = 1e-4      # head에만 적용 (LoRA는 0)
WARMUP_RATIO       = 0.05
NUM_WORKERS        = 4
SEED               = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess(frames):
    """단일 영상 (T,H,W,C) uint8 → (1,C,T,H,W) float ImageNet-normalized.

    extract_videomae_v2.py:preprocess() 와 동일 — 학습/feature 일관성 유지.
    """
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames)
    x = frames.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, IMG_SIZE, IMG_SIZE).permute(0, 2, 1, 3, 4)
    x = (x - MEAN) / STD
    return x


def video_collate(batch):
    """VideoDataset의 4-tuple list → (frames_list, labels_tensor).

    영상 원본 해상도가 다를 수 있어 default_collate가 stack 못함.
    preprocess에서 224 resize 후에 GPU에서 stack.
    """
    frames_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return frames_list, labels


def batch_to_device(frames_list, device):
    """list of frames numpy → (B,C,T,H,W) tensor on device."""
    return torch.cat([preprocess(f) for f in frames_list], dim=0).to(device)


def evaluate(model, loader, loss_fn, device):
    """no_grad + bf16 autocast 평가."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames_list, labels in loader:
            x = batch_to_device(frames_list, device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / total_samples
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(SEED)
    print(f"[setup] device={device}, seed={SEED}")

    # === Data ===
    train_set = VideoDataset(root=f"{DATA_ROOT}/train", num_frames=NUM_FRAMES)
    val_set   = VideoDataset(root=f"{DATA_ROOT}/val",   num_frames=NUM_FRAMES)

    # WeightedRandomSampler — drowning class 적은 불균형 대응
    class_counts = [0] * NUM_CLASSES
    for _, label, _ in train_set.samples:
        class_counts[label] += 1
    print(f"[train] class counts: drowning={class_counts[0]}, normal={class_counts[1]}")

    sample_weights = torch.tensor(
        [1.0 / class_counts[label] for _, label, _ in train_set.samples],
        dtype=torch.float,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_set), replacement=True,
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, collate_fn=video_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=video_collate, pin_memory=True,
    )

    # === Model ===
    model = VideoMAEv2_LoRA_Linear(num_classes=NUM_CLASSES).to(device)

    # === Optimizer — 2 param groups (LoRA / head 다른 lr·wd) ===
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = list(model.head.parameters())
    print(f"[opt] LoRA trainable: {sum(p.numel() for p in lora_params):,}")
    print(f"[opt] Head trainable: {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_LORA, "weight_decay": 0.0},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])

    # === Scheduler (cosine + warmup) ===
    steps_per_epoch = max(1, len(train_loader) // ACCUMULATION_STEPS)
    total_steps  = NUM_EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[sched] total optimizer steps={total_steps}, warmup={warmup_steps}")

    loss_fn = nn.CrossEntropyLoss()

    # === Tracking ===
    best_val_f1, best_epoch, best_state = 0.0, -1, None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    # === Train loop ===
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        optimizer.zero_grad()

        for step, (frames_list, labels) in enumerate(train_loader):
            x = batch_to_device(frames_list, device)
            x.requires_grad_(True)         # reentrant checkpoint 워크어라운드 (lora_backbone.py 참고)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = loss_fn(logits, labels) / ACCUMULATION_STEPS

            loss.backward()
            epoch_loss += loss.item() * ACCUMULATION_STEPS
            num_batches += 1

            if (step + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = epoch_loss / num_batches
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: "
              f"train={avg_train_loss:.4f} val={val_loss:.4f} "
              f"acc={val_acc:.4f} f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.trainable_state_dict().items()}
            print(f"  → New best F1!")

    # === Save best ===
    print(f"\n=== Best: val_f1={best_val_f1:.4f} @ epoch {best_epoch} ===")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "trainable_state": best_state,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "config": {
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "batch_size": BATCH_SIZE, "accumulation_steps": ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS, "lr_lora": LR_LORA, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "seed": SEED, "num_frames": NUM_FRAMES, "img_size": IMG_SIZE,
        },
    }, OUT_CKPT)
    print(f"Saved checkpoint → {OUT_CKPT}")

    # === Loss curve ===
    epochs_x = range(1, NUM_EPOCHS + 1)
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_x, history["train_loss"], label="train_loss")
    axes[0].plot(epochs_x, history["val_loss"], label="val_loss")
    axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"best ({best_epoch})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_x, history["val_f1"], label="val_f1", color="green")
    axes[1].plot(epochs_x, history["val_acc"], label="val_acc", color="orange")
    axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("Val metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=120)
    print(f"Saved loss curve → {OUT_CURVE}")

    # === Final per-class report (best model on val) ===
    model.load_trainable_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, _, _, all_preds, all_labels = evaluate(model, val_loader, loss_fn, device)

    print("\n=== Per-class metrics (best model on val) ===")
    print(classification_report(all_labels, all_preds,
                                target_names=["drowning", "normal"], digits=4))
    print("=== Confusion matrix (행=실제, 열=예측) ===")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
