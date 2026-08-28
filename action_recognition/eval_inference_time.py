"""
LoRA 모델(VideoMAEv2 + LoRA + head)의 클립 1개당 추론(forward) 시간 측정.

- 대상: datasets/test 의 모든 mp4 클립 (VideoDataset, submerged 제외 등 학습/평가와 동일)
- 측정 범위: **모델 forward 만** (백본+head). 비디오 디코딩/전처리/CPU→GPU 전송은 제외.
- 정확한 GPU 타이밍: warmup 후, 매 클립마다 torch.cuda.synchronize() 로 커널 완료를 보장하고
                     time.perf_counter() 로 측정 (GPU는 비동기라 sync 없으면 시간이 의미 없음).
- 결과: 클립별 시간(ms) + 평균/표준편차/min/max 를 엑셀(.xlsx)로 저장.

사용:
    python eval_inference_time.py
    python eval_inference_time.py --split test --warmup 10 --out inference_time_lora.xlsx
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from data.video_dataset import VideoDataset
from models.lora_backbone import VideoMAEv2_LoRA_Linear
from train_lora import preprocess, DATA_ROOT, NUM_FRAMES, NUM_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/lora_qkv_best.pt",
                   help="LoRA + head 체크포인트")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--warmup", type=int, default=10,
                   help="측정 전 버리는 forward 횟수 (CUDA 커널 캐싱/클럭 안정화)")
    p.add_argument("--out", default="inference_time_lora.xlsx",
                   help="결과 엑셀 경로")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def time_forward(model, x, device):
    """모델 forward 1회의 순수 GPU 시간(ms). 앞뒤로 synchronize 로 정확히 측정."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # 학습과 동일 (bf16)
        logits = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pred = int(logits.argmax(dim=1).item())
    return (t1 - t0) * 1000.0, pred  # ms


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, split={args.split}, warmup={args.warmup}")

    # === Dataset (raw mp4 클립) ===
    dataset = VideoDataset(root=f"{DATA_ROOT}/{args.split}", num_frames=NUM_FRAMES)
    if len(dataset) == 0:
        print("[error] 클립이 없습니다."); return

    # === Model ===
    print(f"\n[Loading] {args.ckpt}")
    model = VideoMAEv2_LoRA_Linear(num_classes=NUM_CLASSES, head_ckpt_path=None).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    model.load_trainable_state_dict(state)
    model.eval()
    print(f"  best_epoch={ckpt.get('best_epoch')}, best_val_f1={ckpt.get('best_val_f1')}")

    id2name = VideoDataset.ID2NAME  # {0: drowning, 1: normal}

    with torch.inference_mode():
        # === Warmup (측정 X) ===
        # 첫 클립으로 warmup → CUDA 커널 컴파일/캐싱, GPU 클럭 ramp-up 안정화
        frames0, _, _, _ = dataset[0]
        x0 = preprocess(frames0).to(device)
        for _ in range(args.warmup):
            time_forward(model, x0, device)
        print(f"[warmup] {args.warmup}회 완료\n")

        # === 본 측정 ===
        rows = []
        for i in range(len(dataset)):
            frames, label, video_name, folder = dataset[i]
            # 전처리/전송은 측정 밖 (forward 만 측정)
            x = preprocess(frames).to(device)
            ms, pred = time_forward(model, x, device)
            rows.append({
                "clip": video_name,
                "folder": folder,
                "label": id2name[label],
                "pred": id2name[pred],
                "forward_ms": ms,
            })
            if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
                print(f"  [{i+1:4d}/{len(dataset)}] {video_name}: {ms:.2f} ms")

    # === 통계 ===
    df = pd.DataFrame(rows)
    times = df["forward_ms"].to_numpy()
    summary = pd.DataFrame([{
        "model": "VideoMAEv2 + LoRA(qkv) + head",
        "split": args.split,
        "num_clips": len(times),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "fps_mean": 1000.0 / float(np.mean(times)),
    }])

    print("\n=== 추론시간 요약 (forward only, LoRA) ===")
    print(f"  클립 수:   {summary['num_clips'][0]}")
    print(f"  평균:      {summary['mean_ms'][0]:.2f} ms")
    print(f"  표준편차:  {summary['std_ms'][0]:.2f} ms")
    print(f"  min/max:   {summary['min_ms'][0]:.2f} / {summary['max_ms'][0]:.2f} ms")
    print(f"  ≈ {summary['fps_mean'][0]:.1f} clips/sec")

    # === 엑셀 저장 ===
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df.to_excel(writer, sheet_name="per_clip", index=False)
    print(f"\n[Saved] {args.out}  (sheets: summary, per_clip)")


if __name__ == "__main__":
    main()
