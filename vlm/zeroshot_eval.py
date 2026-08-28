"""
Qwen3.5-4B 제로샷 익수 판단 평가 (파인튜닝 X).

- vLLM OpenAI 호환 서버(별도 conda env `qwen35`)에 video_url(file://)로 클립을 던져
  "익수/정상" 판단을 받고, 기존 분류기와 동일하게 drowning 중심 지표(P/R/F1)를 계산한다.
- 라벨 매핑은 action_recognition VideoDataset 과 동일:
    drowning → drowning(positive),  floating/standing/swimming/complex → normal,  submerged 제외

선행: vLLM 서버가 떠 있어야 함.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/envs/qwen35/bin/vllm serve weights/Qwen3.5-4B \
     --served-model-name Qwen3.5-4B --host 127.0.0.1 --port 8001 \
     --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 --enforce-eager \
     --reasoning-parser qwen3 --allowed-local-media-path <repo> \
     --media-io-kwargs '{"video": {"num_frames": -1}}' --limit-mm-per-prompt '{"video":1,"image":0}'

실행(같은 env 권장):
  /root/miniconda3/envs/qwen35/bin/python vlm/zeroshot_eval.py
"""
import argparse
import csv
import os
import time
from pathlib import Path

from openai import OpenAI

REPO = "/root/workspace/Drowning-Detection-VLM"

# VideoDataset 과 동일한 폴더→라벨 매핑 (이진: drowning vs normal)
FOLDER2LABEL = {
    "drowning": "drowning",
    "floating": "normal",
    "standing": "normal",
    "swimming": "normal",
    "complex":  "normal",
}
EXCLUDE_FOLDERS = {"submerged"}

PROMPTS = {
    "en": (
        "You are analyzing swimming-pool CCTV footage. "
        "Decide whether the person is in a DROWNING emergency "
        "(struggling, flailing, sinking, unable to stay afloat or breathe) "
        "or doing NORMAL activity (swimming, standing, floating, playing). "
        "End your answer with exactly one line: 'Verdict: drowning' or 'Verdict: normal'."
    ),
    "ko": (
        "이 수영장 CCTV 영상 속 사람의 상태를 판단하세요. "
        "물에 빠져 허우적대거나 가라앉는 '익수' 위험 상황이면 익수, "
        "수영/서있기/떠있기 등 정상 활동이면 정상입니다. "
        "마지막 줄에 반드시 '판단: 익수' 또는 '판단: 정상' 형식으로 결론을 적으세요."
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="Qwen3.5-4B")
    p.add_argument("--lang", default="en", choices=["en", "ko"], help="프롬프트 언어")
    p.add_argument("--fps", type=int, default=8, help="비디오 프레임 샘플링 fps")
    p.add_argument("--temp", type=float, default=0.0, help="샘플링 온도 (평가는 0=결정적 권장)")
    p.add_argument("--think", action="store_true", help="thinking 모드 사용(기본 off=instruct)")
    p.add_argument("--out", default="vlm/zeroshot_results.csv")
    return p.parse_args()


def collect_clips(split):
    """datasets/{split}/{folder}/*.mp4 → [(clip_path, folder, gt_label)] (submerged 제외)."""
    root = Path(REPO) / "datasets" / split
    items = []
    for folder_dir in sorted(root.iterdir()):
        if not folder_dir.is_dir():
            continue
        folder = folder_dir.name
        if folder in EXCLUDE_FOLDERS or folder not in FOLDER2LABEL:
            continue
        gt = FOLDER2LABEL[folder]
        for mp4 in sorted(folder_dir.glob("*.mp4")):
            items.append((str(mp4), folder, gt))
    return items


def _pick(s):
    """문자열에서 drowning/normal 키워드(영/한)가 한쪽만 있으면 그 라벨."""
    s = s.lower()
    d = ("익수" in s) or ("drowning" in s) or ("drown" in s)
    n = ("정상" in s) or ("normal" in s)
    if d and not n:
        return "drowning"
    if n and not d:
        return "normal"
    return None


def parse_pred(text):
    """모델 응답 → 'drowning' / 'normal' / None(애매). 영/한 공통.
    thinking 모드는 본문에 두 키워드가 섞이므로 마지막 결론 줄('Verdict:'/'판단:')을 우선.
    """
    if text is None:
        return None
    verdict = None
    for line in text.splitlines():
        low = line.lower()
        if "verdict" in low or "판단" in line:
            verdict = line
    if verdict:
        r = _pick(verdict)
        if r:
            return r
    return _pick(text)  # fallback: 전체 텍스트에서 한쪽만 등장


def classify(client, model, path, prompt, fps, think, temp):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{path}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=4096 if think else 256,
        temperature=temp,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": think},
            "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True},
        },
    )
    return r.choices[0].message.content


def main():
    args = parse_args()
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    prompt = PROMPTS[args.lang]
    clips = collect_clips(args.split)
    print(f"[setup] split={args.split}, clips={len(clips)}, lang={args.lang}, fps={args.fps}, "
          f"temp={args.temp}, mode={'thinking' if args.think else 'instruct'}")
    if not clips:
        print("[error] 클립 없음"); return

    rows = []
    latencies = []
    for i, (path, folder, gt) in enumerate(clips):
        t0 = time.time()
        try:
            resp = classify(client, args.model, path, prompt, args.fps, args.think, args.temp)
            err = ""
        except Exception as e:
            resp, err = None, str(e)[:200]
        dt = time.time() - t0
        latencies.append(dt)
        pred = parse_pred(resp)
        rows.append({
            "clip": os.path.basename(path), "folder": folder,
            "gt": gt, "pred": pred if pred else "UNPARSED",
            "raw": (resp or "")[:120].replace("\n", " "), "sec": round(dt, 2), "error": err,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(clips):
            print(f"  [{i+1:3d}/{len(clips)}] {folder:9s} gt={gt:8s} "
                  f"pred={str(pred):8s} ({dt:.1f}s)")

    # === 지표 (drowning = positive) ===
    # UNPARSED 는 보수적으로 normal 로 처리 (drowning GT면 '놓침'으로 잡힘)
    def pred_label(r):
        return r["pred"] if r["pred"] in ("drowning", "normal") else "normal"

    tp = sum(1 for r in rows if r["gt"] == "drowning" and pred_label(r) == "drowning")
    fn = sum(1 for r in rows if r["gt"] == "drowning" and pred_label(r) == "normal")
    fp = sum(1 for r in rows if r["gt"] == "normal" and pred_label(r) == "drowning")
    tn = sum(1 for r in rows if r["gt"] == "normal" and pred_label(r) == "normal")
    n = len(rows)
    unparsed = sum(1 for r in rows if r["pred"] == "UNPARSED")
    errors = sum(1 for r in rows if r["error"])

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    # macro F1 (drowning + normal)
    nprec = tn / (tn + fn) if (tn + fn) else 0.0
    nrec = tn / (tn + fp) if (tn + fp) else 0.0
    nf1 = 2 * nprec * nrec / (nprec + nrec) if (nprec + nrec) else 0.0
    macro_f1 = (f1 + nf1) / 2

    print("\n=== Qwen3.5-4B 제로샷 결과 ===")
    print(f"  클립수: {n}  (UNPARSED {unparsed}, 에러 {errors})")
    print(f"  평균 추론시간: {sum(latencies)/len(latencies):.2f}s/clip")
    print(f"\n  Confusion (행=실제, 열=예측)")
    print(f"               drowning   normal")
    print(f"    drowning:   {tp:6d}   {fn:6d}")
    print(f"    normal:     {fp:6d}   {tn:6d}")
    print(f"\n  [Drowning] Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")

    # === 저장 ===
    out = Path(REPO) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "folder", "gt", "pred", "raw", "sec", "error"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
