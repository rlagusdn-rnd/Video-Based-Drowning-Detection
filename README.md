# Video-Based Drowning Detection

## 프로젝트 개요
**프로젝트 명**
익수(drowning)사고 조기감지 파이프라인 구현

**프로젝트 기간**
2025년 ~ 2026년

**프로젝트 목적** 
수영장 환경에서 익수자의 불규칙한 익수행동을 실시간으로 탐지하여 익수자의 발생을 신속하게 알려주고, 구조시간 확보하기 위한 안전 관리 시스템 개발


## 핵심 기술 및 아키텍처
**파이프라인 구조**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   RTSP / mp4                                                            │
│    │                                                                    │ 
│    ├─ YOLO ──── 사람 bbox ──┐                                            │
│    │                        ├──────────────  IOU 교차 검증                │
│    └─ SAM2 ─── ID별 mask ───┘                     │                      │
│                                                  ▼                      │
│                                               ID 별 crop                 │
│                                                  │                      │
│                                                  ▼                      │
│                                        Behavior Classification          │
│                                                  │                      │
│                                          drowning / normal              │
│                                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

[Stage 1] 탐지 및 추적
            │  YOLO 사람 탐지 + SAM2 추적 결과 대조
[Stage 2] 객체별 4초 crop 클립 생성
            │  224 x 224 resize, 16 frames
[Stage 3] 익수 / 일반 행동 분류
            │  VideoMAE v2 K710 pretrain + LoRA finetuning
[Demo]    realtime_detection_lora.py — 탐지·추적·분류 통합 데모
```

핵심 스택: YOLO + SAM2 (탐지·추적) → VideoMAE v2 K710 + LoRA finetuning (행동 인식) / VLM (보조 분석)



## 영상 데모
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

## 기술적 구현
**1. YOLO 탐지 - SAM2 마스크 IOU 대조로 ID 유지**
물 일렁임, 수면, 빛반사 등에 의해 객체 ID 유지 어려움
-> segmentation mask결과와 탐지결과 대조로 같은 객체 ID 유지

realtime_detection_lora.py L407-423
```python
# obj_ids : SAM2가 이번 프레임에 추정 중인 전체 객체 ID
# masks_by_id : 그중 마스크 면적이 MIN_MASK_AREA 이상인 것만 남긴 {id:mask}
for oid in obj_ids:
    if oid in masks_by_id:
        obj_bbox = get_bbox_from_mask(masks_by_id[oid]) # mask -> bbox 변환
        matched_yolo = any(calculate_iou(obj_bbox, yolo_bbox) > MASK_DET_IOU_THRESHOLD for yolo_bbox in yolo_bboxes) # yolo bbox와 mask bbox 겹치는지 체크 (수영자세와 bbox간의 차이로인해 낮은 임계값)
        if matched_yolo:
            miss_counts[oid] = 0             # YOLO가 교차 확인 -> 리셋
        else:
            miss_counts[oid] = miss_counts.get(oid, 0) + 1             # sam2만 잡고 YOLO는 놓침 -> 누적
    else:
        miss_counts[oid] = miss_counts.get(oid, 0) + 1

to_remove = {oid for oid, miss in miss_counts.items()
             if miss > LOST_TIMEOUT
             and not crop_state.get(oid, {}).get('is_drowning', False)}

to_remove |= find_duplicate_tracks(masks_by_id, yolo_bboxes, MASK_OVERLAP_THRESHOLD, MASK_DET_IOU_THRESHOLD) # 한사람을 서로다른 두 ID가 중복 추적 체크

```

**2. 추적 객체 집합 변경시 SAM2 초기화**
연속 segmentation 도중 객체 추가, 삭제가 어려움
-> 추적 대상 등장하는 프레임을 첫 프레임으로 등록하고 SAM2 재등록(object id는 유지)

realtime_detection_lora.py L461-477
```python
# to_promote : 기존 Mask에 없는 탐지 대상을 후보 등록 및 PROMOTION_WINDOW(5프레임) 안에서 PROMOTION_THRESHOLD(3회) 이상 잡힌 것만 승격시킨 목록
# to_remove : 연속 미탐(LOST_TIMEOUT 초과) 또는 중복 추적으로 제거할 기존 트랙
if to_promote or to_remove: # 트랙 집합에 변동 있을 시 SAM2 재시딩
    survivor_bboxes = {oid: get_bbox_from_mask(m) for oid, m in masks_by_id.items() if oid not in to_remove}
    sam2_predictor.load_first_frame(frame_rgb) # SAM2 추적 상태 초기화

    for oid in to_remove:
        miss_counts.pop(oid, None)

    for oid, bbox in survivor_bboxes.items(): # obj_id 그대로 넘김
        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=oid, bbox=bbox_arr)

    # 새 ID 발급
    for temp_id, bbox in to_promote:
        new_id = tracker.next_id
        tracker.next_id += 1
        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=new_id, bbox=bbox_arr)
        miss_counts[new_id] = 0
        del candidates[temp_id]
```

**3. 4초 클립 단위 프레임 저장 및 행동분류**
behavior classification을 위한 전처리 과정 & 행동분류 함수 호출

realtime_detection_lora.py L496-508, L295-315
```python
s = crop_state[oid]

# 버퍼가 4초(cycle_frames)를 채웠을 때만 추론
if len(s['frames']) >= cycle_frames:
    s['is_drowning'] = classify_clip(s['frames'], model, device)
    s['frames'] = []                                              # 버퍼 비우고 다음 주기 시작
    s['crop_box_size'] = int(bbox_max_dim * CROP_SIZE_PADDING)    # 크롭 크기는 주기마다만 갱신
    s['cycle_start_frame'] = frame_count

# 크롭 중심은 매 프레임 EMA로 평활
s['cx'] = CROP_EMA_PADDING * cx_raw + (1 - CROP_EMA_PADDING) * s['cx']
s['cy'] = CROP_EMA_PADDING * cy_raw + (1 - CROP_EMA_PADDING) * s['cy']

crop = crop_with_padding(frame, crop_state[oid])
if crop is not None:
    crop_state[oid]['frames'].append(crop)
```
```python
def classify_clip(frames, model, device):
    """4초 buffer의 frames(BGR 224x224 list) → drowning(True/False)."""
    if len(frames) < NUM_FRAMES:
        return False
    clip = np.stack(frames, axis=0)                              # (N, 224, 224, [B,G,R])

    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int) # 16장 균등 간격 샘플링
    clip = clip[idx]
    clip = clip[..., [2, 1, 0]]                                  # OpenCV(BGR) → 모델 입력(RGB)

    x = torch.from_numpy(clip)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0       # (1, 3, 16, 224, 224), 0~1
    x = x.to(device)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device) 

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)                                        # (1, 2)

    return logits.argmax(dim=1).item() == DROWNING
```

## 데이터셋 구축
```
수영장 CCTV 영상 (https://www.youtube.com/@LifeguardRescue)

1. 탐지 모델 학습 데이터셋: 원본 영상 -> 프레임 추출 -> 사람 라벨링(auto & manual labeling) -> 탐지 학습 데이터셋
구축 이유 : 상반식, 머리만 노출된 상태, 튜브가림, 잠수 등 때문에 도메인 데이터 직접 구축

2. 추적 성능평가 데이터셋: 원본 영상 -> 프레임 추출 -> MOT 포맷 GT 수동 라벨링 -> 추적 성능 평가 데이터셋(HOTA)
구축 이유 : 잠수, 겹침 등으로 인한 ID 변경 최소화하는 Track 파이프라인 구현을 위한 평가데이터셋 구축
성능 평가 후보군 : ByteTrack, BoTSORT, YOLO & SAM2, YOLO & CUTIE
(CVAT 로컬 활용, MOT1.1 포맷)

3. 행동인식 학습 및 평가 데이터셋 : 원본 영상 -> 탐지 & 추적 모델 활용하여 객체별 4초 crop 영상 생성 -> 수동 검수 (ID 유지 여부 + 행동 종류) -> 행동 분류 데이터셋
구축 이유 : 객체별 행동을 판단하여 익수위험군을 도출하기 위한 행동분류 모델 학습 용 데이터셋 생성
행동 종류 : 익수행동(Drowning), 서있기(Standing), 걷기(Walking), 수영(Swimming), 떠있기(Floating), 놀기(Playing)
학습 모델 : VideoMAE v2 K710 backbone + LORA

```


## Setup

### Environment

- Python 3.12 

```bash
conda create -n drowning python=3.12 -y
conda activate drowning
pip install -r requirements.txt
```



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
- YOLO — *Ultralytics*
