"""
익수행동 '분석' 스크립트 — vLLM 오프라인 모드 (서버 X, openai X, HTTP X).

목적: 익수 위험 행동의 정의/징후를 Qwen3.5-4B 에게 주고, 영상에 그 행동이 나타나는지
      분석시켜 JSON 형식의 분석 결과를 받는다. (분류기/F1 아님)

- vLLM `LLM` 클래스로 모델을 이 스크립트 안에서 직접 로드 → llm.chat() 로 추론.
  서버도 openai 패키지도 필요 없음. 전부 로컬 GPU에서 동작.
- 샘플링은 Qwen3.5 README 비디오 예시와 동일:
  temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5,
  mm_processor_kwargs={"fps":2,"do_sample_frames":True}
- 출력: 클립별 분석 텍스트를 jsonl + 사람이 읽는 txt 로 저장.

주의: 반드시 qwen35 env 로 실행 + __main__ 가드 필요(vLLM은 spawn 멀티프로세싱 사용).
실행:
  /root/miniconda3/envs/qwen35/bin/python vlm/analyze_drowning.py --folders drowning --limit 5
"""
import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams

REPO = "/root/workspace/Drowning-Detection-VLM"
MODEL_DIR = f"{REPO}/weights/Qwen3.5-4B"
EXCLUDE_FOLDERS = {"submerged"}

# === 익수 위험 행동 분석 프롬프트 (+ JSON 출력 스키마) ===
PROMPT = """You are a video-understanding model that analyzes drowning-risk behavior in swimming-pool CCTV footage. Observe the given video (or sequence of frames) in temporal order and judge whether the person is close to an actual drowning-risk state. Focus on these visual signs: Does the person's head or face repeatedly sink below the water and come back up? Does the person appear to struggle to keep their face above the water? Are the arms thrashing frantically rather than performing regular swimming strokes? Is the water around the person splashing abnormally or violently disturbed? Do the body movements look like an in-place struggle to survive rather than normal forward-swimming motion?

Answer ONLY in the following JSON format.
{
  "risk_level": "low / medium / high",
  "event_type": "normal_swimming / playing / possible_drowning / unclear",
  "drowning_signs": {
    "head_submergence_repetition": "yes / no / unclear",
    "struggling_to_keep_face_above_water": "yes / no / unclear",
    "frantic_arm_movement": "yes / no / unclear",
    "abnormal_water_splash": "yes / no / unclear",
    "lack_of_forward_progression": "yes / no / unclear"
  },
  "key_evidence": [ "Write 2-4 key pieces of visual evidence observed in the video" ],
  "time_interval": "The time interval where the risk signs are clearest, or unclear",
  "final_judgement": "Write the final judgement in one sentence"
}"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--folders", default="", help="콤마구분 폴더 필터 (예: drowning). 비우면 전체")
    p.add_argument("--limit", type=int, default=0, help="최대 클립 수 (0=전체)")
    p.add_argument("--fps", type=int, default=2, help="비디오 샘플링 fps (예시 기본 2)")
    p.add_argument("--think", action="store_true", help="thinking 모드 (기본 off=instruct)")
    p.add_argument("--out", default="vlm/analysis.jsonl")
    return p.parse_args()


def collect_clips(split, folders_filter):
    root = Path(REPO) / "datasets" / split
    want = {f.strip() for f in folders_filter.split(",") if f.strip()}
    items = []
    for folder_dir in sorted(root.iterdir()):
        if not folder_dir.is_dir() or folder_dir.name in EXCLUDE_FOLDERS:
            continue
        if want and folder_dir.name not in want:
            continue
        for mp4 in sorted(folder_dir.glob("*.mp4")):
            items.append((str(mp4), folder_dir.name))
    return items


def _last_balanced_object(text):
    """마지막 '}'에서 거꾸로 중괄호 균형을 맞춰 최상위 {...} 블록을 찾는다."""
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i:end + 1]
    return None


def extract_json(text):
    """응답에서 최종 JSON만 추출 (thinking 중괄호 / ```json 펜스 / 중첩객체 대응)."""
    if not text:
        return None
    import re
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    bal = _last_balanced_object(text)
    if bal:
        candidates.append(bal)
    for c in reversed(candidates):  # 마지막(결론) JSON 우선
        try:
            return json.loads(c.strip())
        except json.JSONDecodeError:
            continue
    return None


def main():
    args = parse_args()

    clips = collect_clips(args.split, args.folders)
    if args.limit > 0:
        clips = clips[:args.limit]
    print(f"[setup] split={args.split} clips={len(clips)} folders={args.folders or 'ALL'} "
          f"fps={args.fps} mode={'thinking' if args.think else 'instruct'}")
    if not clips:
        print("[error] 클립 없음"); return

    # === vLLM 오프라인 모델 로드 (이 프로세스 안에서 GPU에 직접 올림) ===
    t0 = time.time()
    llm = LLM(
        model=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        enforce_eager=True,                       # 16GB OOM 회피
        limit_mm_per_prompt={"video": 1, "image": 0},
        allowed_local_media_path=REPO,            # file:// 로컬 mp4 허용
        media_io_kwargs={"video": {"num_frames": -1}},  # fps 제어 허용
    )
    print(f"[model loaded] {time.time()-t0:.1f}s")

    # README 비디오 예시와 동일한 샘플링
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20,
                        presence_penalty=1.5, max_tokens=8192)

    out_jsonl = Path(REPO) / args.out
    out_txt = out_jsonl.with_suffix(".txt")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(out_txt, "w") as ftxt:
        for i, (path, folder) in enumerate(clips):
            name = os.path.basename(path)
            messages = [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{path}"}},
                {"type": "text", "text": PROMPT},
            ]}]
            t1 = time.time()
            try:
                out = llm.chat(messages, sampling_params=sp,
                               chat_template_kwargs={"enable_thinking": args.think},
                               mm_processor_kwargs={"fps": args.fps, "do_sample_frames": True})
                text = out[0].outputs[0].text
                err = ""
            except Exception as e:
                text, err = "", str(e)[:200]
            dt = time.time() - t1

            parsed = extract_json(text)
            records.append({"clip": name, "folder": folder, "analysis_raw": text,
                            "analysis_json": parsed, "sec": round(dt, 2), "error": err})

            block = (f"\n{'='*78}\n[{i+1}/{len(clips)}] {name}  (folder={folder}, {dt:.1f}s)\n"
                     f"{'-'*78}\n{text or '[ERROR] ' + err}\n")
            print(block)
            ftxt.write(block); ftxt.flush()

    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[Saved] {out_jsonl}\n[Saved] {out_txt}")


if __name__ == "__main__":
    main()
