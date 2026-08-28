# Video-Based Drowning Detection

수영장 CCTV 영상에서 **익수(drowning) 사고를 조기 감지**하는 비전 파이프라인.
실시간 RTSP 스트림에서 익수자를 탐지·추적하고, 행동 인식 모델로 위험 행동 분류

> 목표: 실시간 스트림에서 익수자 탐지 → 경보
> 핵심 스택: YOLO + SAM2 (탐지·추적) → VideoMAE v2 K710 + LoRA finetuning (행동 인식) / VLM (보조 분석)

---

## Pipeline

```
[Stage 0] 원본 영상 / RTSP 스트림
            │  YOLO 사람 탐지 + SAM2 추적
[Stage 1] 객체별 4초 crop 클립 생성
            │
[Stage 2] VideoMAE v2 K710 pretrain + LoRA finetuning → 행동 분류
            │
[Demo]    realtime_detection_lora.py — 탐지·추적·분류 통합 데모
```

---



https://github.com/user-attachments/assets/adac6fdd-c303-48cb-aeaf-0d1077ad5fe8



## Directory Structure

```
Video-Based-Drowning-Detection/
│
├── 탐지 · 추적 (YOLO + SAM2 streaming)
│   ├── test_sam2_track.py            # 추적 파이프라인 (순수)
│   ├── test_sam2_track_roi.py        # 추적 + ROI 영역 제한
│   ├── test_sam2_rtsp_roi_crop.py    # 추적 + ROI + crop 클립 자동 저장 (학습데이터 생성)
│   ├── test_sam2_crop_object.py      # 익수자 수동 선택 → 클립화 도구
│   └── roi_make.py                   # ROI 폴리곤 라벨링 도구
│
├── 실시간 데모
│   ├── realtime_detection_lora.py    # 최종 LoRA 모델 통합 데모
│   └── realtime_detection.py         # finetune 버전 (비교용)
│
├── action_recognition/               # 행동 인식 (VideoMAE v2 + LoRA)
│   ├── train_lora.py                 # LoRA 학습
│   ├── test_f1_lora.py               # per-class P/R/F1 평가
│   ├── eval_test_data.py             # 2-class 평가
│   ├── eval_inference_time.py        # 추론 속도 측정
│   ├── models/                       # head.py, lora_backbone.py
│   └── data/                         # video_dataset.py (mp4 → 16-frame)
│
├── vlm/                              # VLM 추론 (별도 환경)
│   ├── zeroshot_eval.py              # Qwen VLM 제로샷 익수 판단 평가
│   └── analyze_drowning.py           # vLLM 오프라인 익수행동 분석
│
├── requirements.txt
└── .gitignore
```

## Setup

### Environment

- Python 3.10 

```bash
conda create -n drowning python=3.10 -y
conda activate drowning
pip install -r requirements.txt
```


### 가중치 / 체크포인트

직접 학습한 가중치 2개 (Object detection, action classification model) 저장소에 포함


### 데이터셋 레이아웃

```
datasets/
├── train/{class}/*.mp4     # 4초 crop 클립
├── val/{class}/*.mp4
└── test/{class}/*.mp4
```


## References

- SAM 2 — *Segment Anything in Images and Videos* (Meta AI)
- VideoMAE v2 — *Scaling Video Masked Autoencoders with Dual Masking* (CVPR 2023)
- LoRA — *Low-Rank Adaptation of Large Language Models*
- Cutie — *Putting the Object Back into Video Object Segmentation* (CVPR 2024)
