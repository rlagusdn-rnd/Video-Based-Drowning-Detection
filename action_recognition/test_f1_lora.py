"""
LoRA fine-tuned 모델 test 평가.

학습된 LoRA + head checkpoint 로드 → test split에서 per-class P/R/F1 + confusion matrix.
preprocess / video_collate / evaluate 는 train_lora.py 의 것 재사용 (학습/평가 일관성).

사용:
    python test_f1_lora.py                                    # default: test split, lora_qkv_best.pt
    python test_f1_lora.py --split val
    python test_f1_lora.py --ckpt checkpoints/other.pt
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from data.video_dataset import VideoDataset
from models.lora_backbone import VideoMAEv2_LoRA_Linear
from train_lora import (
    DATA_ROOT, NUM_FRAMES, NUM_CLASSES, NUM_WORKERS,
    video_collate, evaluate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lora_qkv_best.pt",
                   help="LoRA + head 체크포인트 경로")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=4,
                   help="평가는 backward 없으니 학습보다 크게 잡을 수 있음")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, split={args.split}, batch={args.batch_size}")

    # === Dataset ===
    test_set = VideoDataset(root=f"{DATA_ROOT}/{args.split}", num_frames=NUM_FRAMES)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=video_collate, pin_memory=True,
    )

    # === Checkpoint ===
    print(f"\n[Loading] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  best_epoch:  {ckpt.get('best_epoch', 'N/A')}")
    best_val_f1 = ckpt.get("best_val_f1")
    if best_val_f1 is not None:
        print(f"  best_val_f1: {best_val_f1:.4f}")
    cfg = ckpt.get("config", {})
    if cfg:
        print(f"  config: r={cfg.get('lora_r')}, alpha={cfg.get('lora_alpha')}, "
              f"epochs={cfg.get('num_epochs')}, lr_lora={cfg.get('lr_lora')}")

    # === Model ===
    # head_ckpt_path=None: 어차피 trainable_state로 덮어쓸 거라 best64 로드 skip
    model = VideoMAEv2_LoRA_Linear(num_classes=NUM_CLASSES, head_ckpt_path=None).to(device)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    model.load_trainable_state_dict(state)
    model.eval()

    # === Evaluate ===
    loss_fn = nn.CrossEntropyLoss()
    avg_loss, accuracy, macro_f1, all_preds, all_labels = evaluate(
        model, test_loader, loss_fn, device
    )
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    # === Summary ===
    print(f"\n=== {args.split} 결과 (LoRA + head) ===")
    print(f"Loss:        {avg_loss:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # Val vs Test 비교 (val에 과적합 진단)
    if best_val_f1 is not None and args.split == "test":
        gap = best_val_f1 - macro_f1
        diagnosis = "(val 과적합 의심)" if gap > 0.05 else "(일반화 양호)"
        print(f"\n  Val Macro F1:    {best_val_f1:.4f}")
        print(f"  Test Macro F1:   {macro_f1:.4f}")
        print(f"  Gap (val-test):  {gap:+.4f}  {diagnosis}")

    # === Per-class ===
    target_names = ["drowning", "normal"]
    print("\n=== Per-class metrics ===")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names, digits=4))

    # === Confusion matrix ===
    print("=== Confusion matrix (행=실제, 열=예측) ===")
    cm = confusion_matrix(all_labels, all_preds)
    header = "               " + "  ".join(f"{n:>10}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>10}:  " + "  ".join(f"{v:>10d}" for v in row))

    # === Drowning class 단독 강조 (우선순위) ===
    print("\n=== Drowning class 단독 (우선순위 지표) ===")
    drown_tp = cm[0, 0]
    drown_fn = cm[0, 1]
    drown_fp = cm[1, 0]
    drown_tn = cm[1, 1]
    print(f"  TP={drown_tp}, FN={drown_fn} (놓친 익수), "
          f"FP={drown_fp} (오경보), TN={drown_tn}")
    print(f"  Recall   (놓침 없는 정도): {drown_tp / (drown_tp + drown_fn):.4f}")
    print(f"  Precision (오경보 적음):  {drown_tp / (drown_tp + drown_fp):.4f}"
          if (drown_tp + drown_fp) > 0 else "  Precision: N/A")


if __name__ == "__main__":
    main()
