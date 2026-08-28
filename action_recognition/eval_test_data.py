"""
test_data(100개 객체 crop 클립)를 LoRA 모델(VideoMAEv2 + LoRA(qkv) + head)로
추론하고 2-class(drowning/normal) 분류 성능을 평가/저장한다.

- 입력: datasets/test_data/{abnormal,imitation,normal}/*.mp4
        (이미 224x224 객체 crop 클립 — 파이프라인 Stage1 산출물과 동일 포맷이라
         YOLO+SAM2 재실행 없이 바로 모델에 forward)
- 라벨 매핑(사용자 결정):
        abnormal  -> drowning(0)   # 실제 익수
        imitation -> drowning(0)   # 익수 흉내도 잡아야 할 positive
        normal    -> normal(1)
- 전처리/forward: train_lora.preprocess() 와 동일(16프레임 균등샘플, ImageNet 정규화, bf16).
- 평가 우선순위: drowning(positive) precision / recall / F1.
- 결과: 콘솔 출력 + 엑셀(summary / per_folder / confusion / per_clip).

사용:
    python eval_test_data.py
    python eval_test_data.py --data datasets/test_data --out eval_test_data_lora.xlsx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from models.lora_backbone import VideoMAEv2_LoRA_Linear
from train_lora import preprocess, NUM_FRAMES, NUM_CLASSES

# test_data 폴더명 -> 2-class 라벨 ID
FOLDER2ID = {
    "abnormal": 0,   # drowning
    "imitation": 0,  # drowning (익수 흉내도 positive)
    "normal": 1,     # normal
}
ID2NAME = {0: "drowning", 1: "normal"}
DROWNING = 0  # positive class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/root/workspace/Drowning-Detection-VLM/datasets/test_data",
                   help="test_data 루트 (하위에 abnormal/imitation/normal 폴더)")
    p.add_argument("--ckpt", default="checkpoints/lora_qkv_best.pt",
                   help="LoRA + head 체크포인트")
    p.add_argument("--out", default="eval_test_data_lora.xlsx", help="결과 엑셀 경로")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def collect_clips(data_root):
    """(path, label_id, folder) 리스트. FOLDER2ID에 있는 폴더만, 파일명 정렬."""
    root = Path(data_root)
    samples = []
    for folder, label in FOLDER2ID.items():
        folder_dir = root / folder
        if not folder_dir.is_dir():
            print(f"[warn] 폴더 없음: {folder_dir}")
            continue
        for vp in sorted(folder_dir.glob("*.mp4")):
            samples.append((vp, label, folder))
    return samples


def load_clip_frames(path, num_frames):
    """mp4 -> (num_frames, H, W, 3) uint8, 균등 샘플."""
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames).astype(int)
    return vr.get_batch(idx.tolist()).asnumpy()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, data={args.data}")

    samples = collect_clips(args.data)
    if not samples:
        print("[error] 클립이 없습니다.")
        return
    print(f"[data] 총 {len(samples)}개 클립")
    for folder in FOLDER2ID:
        n = sum(1 for _, _, f in samples if f == folder)
        print(f"  └ {folder} -> {ID2NAME[FOLDER2ID[folder]]}: {n}")

    # === 모델 로드 (eval_inference_time.py 와 동일) ===
    print(f"\n[loading] {args.ckpt}")
    model = VideoMAEv2_LoRA_Linear(num_classes=NUM_CLASSES, head_ckpt_path=None).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    model.load_trainable_state_dict(state)
    model.eval()
    print(f"  best_epoch={ckpt.get('best_epoch')}, best_val_f1={ckpt.get('best_val_f1')}")

    # === 추론 ===
    rows = []
    with torch.inference_mode():
        for i, (path, label, folder) in enumerate(samples):
            frames = load_clip_frames(path, NUM_FRAMES)
            x = preprocess(frames).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)[0]
            pred = int(logits.argmax(dim=1).item())
            rows.append({
                "clip": path.stem,
                "folder": folder,
                "label_id": label,
                "label": ID2NAME[label],
                "pred_id": pred,
                "pred": ID2NAME[pred],
                "correct": int(pred == label),
                "p_drowning": float(prob[DROWNING]),
                "p_normal": float(prob[1]),
            })
            if (i + 1) % 20 == 0 or (i + 1) == len(samples):
                print(f"  [{i+1:3d}/{len(samples)}] {path.stem}: pred={ID2NAME[pred]} (gt={ID2NAME[label]})")

    df = pd.DataFrame(rows)
    y_true = df["label_id"].to_numpy()
    y_pred = df["pred_id"].to_numpy()

    # === 지표 (positive=drowning) ===
    # labels=[0,1] 고정 → pos_label=0(drowning) 지표를 명시적으로 뽑는다.
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 행=실제, 열=예측
    # drowning 기준 TP/FP/FN/TN
    tp = int(cm[0, 0]); fn = int(cm[0, 1]); fp = int(cm[1, 0]); tn = int(cm[1, 1])

    print("\n=== test_data 평가 (positive=drowning) ===")
    print(f"  전체 정확도(accuracy): {acc*100:.2f}%  ({int((y_true==y_pred).sum())}/{len(df)})")
    print(f"  [drowning] precision={p[0]*100:.2f}%  recall={r[0]*100:.2f}%  f1={f1[0]*100:.2f}%  (support={int(sup[0])})")
    print(f"  [normal  ] precision={p[1]*100:.2f}%  recall={r[1]*100:.2f}%  f1={f1[1]*100:.2f}%  (support={int(sup[1])})")
    print(f"  혼동행렬(행=실제, 열=예측, 순서 [drowning, normal]):")
    print(f"      실제\\예측   drowning  normal")
    print(f"      drowning      {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"      normal        {cm[1,0]:5d}   {cm[1,1]:5d}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    # === 폴더별 분해 (abnormal/imitation/normal 각각 정확도) ===
    folder_rows = []
    for folder in FOLDER2ID:
        sub = df[df["folder"] == folder]
        if len(sub) == 0:
            continue
        gt_name = ID2NAME[FOLDER2ID[folder]]
        n_drown_pred = int((sub["pred_id"] == DROWNING).sum())
        folder_rows.append({
            "folder": folder,
            "gt_class": gt_name,
            "n_clips": len(sub),
            "n_correct": int(sub["correct"].sum()),
            "accuracy_%": round(float(sub["correct"].mean()) * 100, 2),
            "pred_drowning": n_drown_pred,
            "pred_normal": len(sub) - n_drown_pred,
        })
    folder_df = pd.DataFrame(folder_rows)
    print("\n=== 폴더별 분해 ===")
    print(folder_df.to_string(index=False))

    # === 저장 ===
    summary_df = pd.DataFrame([{
        "model": "VideoMAEv2 + LoRA(qkv) + head",
        "num_clips": len(df),
        "accuracy_%": round(acc * 100, 2),
        "drowning_precision_%": round(float(p[0]) * 100, 2),
        "drowning_recall_%": round(float(r[0]) * 100, 2),
        "drowning_f1_%": round(float(f1[0]) * 100, 2),
        "normal_precision_%": round(float(p[1]) * 100, 2),
        "normal_recall_%": round(float(r[1]) * 100, 2),
        "normal_f1_%": round(float(f1[1]) * 100, 2),
        "TP_drowning": tp, "FP_drowning": fp, "FN_drowning": fn, "TN_drowning": tn,
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_f1": ckpt.get("best_val_f1"),
    }])
    cm_df = pd.DataFrame(cm, index=["actual_drowning", "actual_normal"],
                         columns=["pred_drowning", "pred_normal"])

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        folder_df.to_excel(writer, sheet_name="per_folder", index=False)
        cm_df.to_excel(writer, sheet_name="confusion")
        df.to_excel(writer, sheet_name="per_clip", index=False)
    print(f"\n[saved] {args.out}  (sheets: summary, per_folder, confusion, per_clip)")


if __name__ == "__main__":
    main()
