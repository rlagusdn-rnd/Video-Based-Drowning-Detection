"""
realtime_detection.py 의 LoRA 버전.

원본은 finetune 백본 + head_finetune_best64.pt (두 모듈 수동 연결)를 사용했지만,
이 파일은 성능이 더 좋았던 LoRA 모델(VideoMAEv2 + LoRA(qkv) + head, 단일 nn.Module)을 사용한다.

차이점만 요약:
  - load_videomae_backbone(...) + 별도 head  →  VideoMAEv2_LoRA_Linear 단일 모델
  - classify_clip: head(backbone(x))  →  model(x), autocast fp16 → bf16(학습과 동일)
  - 여러 영상(swim_70, swim_71)을 순차 처리해서 데모 mp4 생성
"""
import cv2
import numpy as np
import torch
import sys, os
import threading
import time
import json

from pathlib import Path
from queue import Queue
from hydra import initialize_config_dir

# action_recognition 를 path 에 추가해야 lora_backbone 의 `from models.head import ...` 가 동작
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'action_recognition'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'sam2_streaming'))

from models.lora_backbone import VideoMAEv2_LoRA_Linear

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_camera_predictor

from hydra.core.global_hydra import GlobalHydra
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

from ultralytics import YOLO

# 데모로 뽑을 입력 영상들 (순차 처리)
VIDEO_PATHS = [
    'datasets/videos/swim_70.mp4',
    'datasets/videos/swim_71.mp4',
]
SHOW = False  # True면 cv2.imshow 로 실시간 표시(디스플레이 필요). 헤드리스 저장만 할 땐 False.

yolo_weight_path = 'weights/yolo/swim_0627.pt'
SAM2_CHECKPOINT  = 'sam2/checkpoints/sam2.1_hiera_small.pt'
SAM2_MODEL_CFG   = 'sam2.1/sam2.1_hiera_s.yaml'

LORA_CKPT = 'action_recognition/checkpoints/lora_qkv_best.pt'

OUT_SUFFIX = '_lora_keepid_t60'  # 출력 파일 접미사 (method A + LOST_TIMEOUT=60 실험)

DET_DET_IOU_THRESHOLD = 0.3         # 이전/현재 프레임 DET 결과 비교 [기존 객체 후보와 신규 yolo 검출 결과 매칭에 사용]
MASK_OVERLAP_THRESHOLD = 0.8        # SAM2 마스크 중복 판단 임계값 [중복 추적 감지에 사용]
MASK_DET_IOU_THRESHOLD = 0.15       # SAM2 MASK BBOX / YOLO DET 결과 비교 [추적 객체의 사람 판단 / 신규 객체 후보 등록 시 기존 객체와의 매칭 판단에 사용]
MASK_DET_IOU_GATE_THRESHOLD = 0.15  # SAM2 MASK BBOX / YOLO DET 결과 비교 [신규 객체 후보 등록 시 기존 SAM2 객체와의 매칭 판단에 사용]

PROMOTION_THRESHOLD = 3             # 새 객체로 승격하기 위한 최소 매칭 프레임 수 (PROMOTION_WINDOW 내에서 True인 프레임 수)
PROMOTION_WINDOW = 5                # 객체 승격 후보 검증 프레임 수
LOST_TIMEOUT = 60                   # 객체를 잃었다고 판단하기 위한 연속 불일치 프레임 수 (→ 61프레임/약 2.0초에서 삭제)
CANDIDATE_MAX_UNSEEN_FRAMES = 10    # 객체 승격 후보가 너무 오래 보이지 않는 경우 제거하기 위한 프레임 수
MIN_MASK_AREA = 60                  # SAM2 마스크 최소 면적 기준(픽셀 값)

CROP_CYCLE_SECONDS = 4              # 행동분류 추론 주기
CROP_OUTPUT_SIZE = 224             # 비디오 백본 입력 해상도
CROP_SIZE_PADDING = 1.8            # CROP ROI 비율
CROP_EMA_PADDING = 0.3            # EMA 변수
NUM_FRAMES = 16                     # 비디오 백본 입력 프레임 수
NUM_CLASSES = 2                     # 분류 CLASS (DROWNING VS NORMAL)
DROWNING = 0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1) # videomae 정규화
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


class VideoReader:
    def __init__(self, video_path, queue_size=32, realtime=False):
        """
        realtime=True  : 실시간(RTSP) 모드. 30fps 페이스로 읽고, 큐가 차면
                         오래된 프레임을 버려 최신 프레임을 따라잡음(프레임 손실 O).
        realtime=False : 검증(파일) 모드. 프레임을 버리지 않고 큐가 빌 때까지
                         대기(backpressure) → 모든 프레임 처리(느리지만 손실 X).
        """
        self.video_path = video_path
        self.frame_queue = Queue(maxsize=queue_size)
        self.stop_flag = False
        self.thread = None
        self.realtime = realtime

    def start(self):
        self.thread = threading.Thread(target=self._read_frames)
        self.thread.daemon = True
        self.thread.start()

    def _read_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        try:
            while not self.stop_flag:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.realtime:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put((ret, frame))
                    time.sleep(0.033)
                else:
                    self.frame_queue.put((ret, frame))
        finally:
            cap.release()
            self.frame_queue.put((False, None))

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stop_flag = True
        print("쓰레드 중단")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TrackManager:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 1


class Candidate:
    """SAM2에 등록하기 전 객체 후보 정보 저장 클래스"""
    def __init__(self, bbox, frame_idx):
        self.bbox = bbox
        self.history = [True]
        self.last_seen = frame_idx


def calculate_iou(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def calculate_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return inter / (min(area1, area2) + 1e-6)


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    return bbox


def get_centerpoint_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return (center_x, center_y)


def color_for_id(obj_id):
    """obj_id로부터 결정적인 BGR 색깔을 만든다 (시각화용)."""
    hue = (obj_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def find_duplicate_tracks(masks_by_id: dict, yolo_bboxes: list, mask_overlap_threshold: float, mask_det_iou_threshold: float) -> set:
    """mask overlap + YOLO 매칭으로 중복 트랙 감지."""
    yolo_matched = {}
    for oid, mask in masks_by_id.items():
        bbox = get_bbox_from_mask(mask)
        yolo_matched[oid] = any(calculate_iou(bbox, det) > mask_det_iou_threshold for det in yolo_bboxes)

    to_remove_local = set()
    oids_list = list(masks_by_id.keys())
    for i in range(len(oids_list)):
        a = oids_list[i]
        if a in to_remove_local:
            continue
        for j in range(i + 1, len(oids_list)):
            b = oids_list[j]
            if b in to_remove_local:
                continue

            mask_a = masks_by_id[a]
            mask_b = masks_by_id[b]
            inter = int(np.logical_and(mask_a, mask_b).sum())
            if inter == 0:
                continue

            area_a = int(mask_a.sum())
            area_b = int(mask_b.sum())
            smaller = min(area_a, area_b)
            mask_overlap = inter / (smaller + 1e-6)
            if mask_overlap <= mask_overlap_threshold:
                continue

            a_is_person = yolo_matched[a]
            b_is_person = yolo_matched[b]

            if a_is_person and not b_is_person:
                to_remove_local.add(b)
            elif b_is_person and not a_is_person:
                to_remove_local.add(a)
            else:
                loser = a if area_a < area_b else b
                to_remove_local.add(loser)
    return to_remove_local


def get_sam2_masks(predictor: SAM2ImagePredictor, frame_rgb: np.ndarray, yolo_bboxes: list) -> list:
    if not yolo_bboxes:
        return []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor.set_image(frame_rgb)
        input_boxes = np.array(yolo_bboxes, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=input_boxes, multimask_output=False,)
        if masks.ndim == 3:
            masks = masks[np.newaxis]
    return [masks[i, 0].astype(bool) for i in range(len(yolo_bboxes))]


def crop_with_padding(frame, smooth_state, output_size=CROP_OUTPUT_SIZE):
    """smooth_state 기준 정사각형 crop. 영상 밖은 검정 패딩 + output_size로 resize."""
    cx = int(smooth_state['cx'])
    cy = int(smooth_state['cy'])
    size = smooth_state['crop_box_size']
    half = size // 2

    H, W = frame.shape[:2]
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return cv2.resize(crop, (output_size, output_size))


def load_lora_model(device):
    """VideoMAEv2 + LoRA(qkv) + head 단일 모델 로드 (lora_qkv_best.pt)."""
    model = VideoMAEv2_LoRA_Linear(num_classes=NUM_CLASSES, head_ckpt_path=None).to(device)
    ckpt = torch.load(LORA_CKPT, map_location='cpu', weights_only=False)
    state = {k: v.to(device) for k, v in ckpt["trainable_state"].items()}
    model.load_trainable_state_dict(state)
    model.eval()
    print(f"[LoRA loaded] best_epoch={ckpt.get('best_epoch')}, best_val_f1={ckpt.get('best_val_f1')}")
    return model


def classify_clip(frames, model, device):
    """4초 buffer의 frames(BGR 224x224 list) → drowning(True/False).
    LoRA 단일 모델: logits = model(x). 학습과 동일하게 bf16 autocast 사용.
    """
    if len(frames) < NUM_FRAMES:
        return False

    clip = np.stack(frames, axis=0)                              # (N, 224, 224, [B,G,R])
    idx = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
    clip = clip[idx]
    clip = clip[..., [2, 1, 0]]                                  # BGR -> RGB

    x = torch.from_numpy(clip)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0       # (1, 3, 16, 224, 224), 0~1
    x = x.to(device)
    x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device) # ImageNet 정규화 (학습 preprocess와 동일)

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)                                        # (1, 2)

    return logits.argmax(dim=1).item() == DROWNING


def load_roi(video_path):
    roi_path = Path(video_path).with_suffix('.json')
    if not roi_path.is_file():
        print(f"ROI 파일이 없습니다: {roi_path}")
        return None
    with open(roi_path) as f:
        pts = json.load(f)["points"]
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def in_roi(bbox, polygon):
    if polygon is None:
        return True
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    return cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0


def create_video_writer(video_path, frame, fps=30.0, out_dir='outputs'):
    """추론 결과를 저장할 cv2.VideoWriter 생성. 출력: outputs/{영상이름}_lora_detected.mp4"""
    h, w = frame.shape[:2]
    out_path = Path(out_dir) / f"{Path(video_path).stem}{OUT_SUFFIX}_detected.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    print(f"[Save] {out_path} ({w}x{h} @ {fps}fps)")
    return writer


@torch.inference_mode()
@torch.amp.autocast('cuda')
def run_demo(video_path, yolo_model, model, sam2_predictor, device):
    """단일 영상에 대해 YOLO+SAM2 추적 + LoRA 익수 분류 + bbox 시각화 → mp4 저장."""
    if not os.path.isfile(video_path):
        print(f"there is no video file at {video_path}")
        return

    FPS = 30
    cycle_frames = int(FPS * CROP_CYCLE_SECONDS)
    roi_polygon = load_roi(video_path)

    tracker = TrackManager()
    crop_state = {}
    frame_count = 0
    yolo_bboxes = []
    miss_counts = {}
    candidates = {}
    next_temp_id = 1

    with VideoReader(video_path, realtime=False) as video_reader:
        time.sleep(0.5)
        out = None

        while True:
            t_start = time.time()
            ret, frame = video_reader.read()
            if not ret:
                print(f"영상 종료: {video_path}, frame_count={frame_count}")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            if frame_count == 1:
                sam2_predictor.load_first_frame(frame_rgb)
                results = yolo_model.predict(frame, verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    if not in_roi([x1, y1, x2, y2], roi_polygon):
                        continue
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tracker.next_id, bbox=bbox)
                    tracker.next_id += 1
            else:
                obj_ids, mask_logits = sam2_predictor.track(frame_rgb)
                masks_by_id = {}

                masks_all = (mask_logits > 0.0).squeeze(1).cpu().numpy()
                for i, oid in enumerate(obj_ids):
                    if masks_all[i].sum() >= MIN_MASK_AREA:
                        masks_by_id[oid] = masks_all[i]

                results = yolo_model.predict(frame, verbose=False)
                yolo_bboxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    if not in_roi([x1, y1, x2, y2], roi_polygon):
                        continue
                    yolo_bboxes.append([x1, y1, x2, y2])

                for oid in obj_ids:
                    if oid in masks_by_id:
                        obj_bbox = get_bbox_from_mask(masks_by_id[oid])
                        matched_yolo = any(calculate_iou(obj_bbox, yolo_bbox) > MASK_DET_IOU_THRESHOLD for yolo_bbox in yolo_bboxes)
                        if matched_yolo:
                            miss_counts[oid] = 0
                        else:
                            miss_counts[oid] = miss_counts.get(oid, 0) + 1
                    else:
                        miss_counts[oid] = miss_counts.get(oid, 0) + 1
                # method A: 이미 drowning으로 판정된 트랙은 YOLO miss로 제거하지 않음
                # (트랙이 안 죽으니 ID 유지 → crop_state[oid]['is_drowning'] 유지 → 빨간 박스 안 풀림)
                to_remove = {oid for oid, miss in miss_counts.items()
                             if miss > LOST_TIMEOUT
                             and not crop_state.get(oid, {}).get('is_drowning', False)}

                to_remove |= find_duplicate_tracks(masks_by_id, yolo_bboxes, MASK_OVERLAP_THRESHOLD, MASK_DET_IOU_THRESHOLD)

                matched_candidates_ids = set()
                for det_bbox in yolo_bboxes:
                    is_existing = any(calculate_iou(det_bbox, get_bbox_from_mask(m)) > MASK_DET_IOU_GATE_THRESHOLD for m in masks_by_id.values())
                    if is_existing:
                        continue

                    matched = False
                    for tid, cand in candidates.items():
                        if calculate_iou(det_bbox, cand.bbox) > DET_DET_IOU_THRESHOLD:
                            cand.history.append(True)
                            cand.bbox = det_bbox
                            cand.last_seen = frame_count
                            matched_candidates_ids.add(tid)
                            matched = True
                            break

                    if not matched:
                        candidates[next_temp_id] = Candidate(det_bbox, frame_count)
                        matched_candidates_ids.add(next_temp_id)
                        next_temp_id += 1

                for tid, cand in candidates.items():
                    if tid not in matched_candidates_ids:
                        cand.history.append(False)

                for cand in candidates.values():
                    if len(cand.history) > PROMOTION_WINDOW:
                        cand.history = cand.history[-PROMOTION_WINDOW:]

                candidates = {tid: cand for tid, cand in candidates.items() if frame_count - cand.last_seen <= CANDIDATE_MAX_UNSEEN_FRAMES}

                to_promote = []
                for tid, cand in candidates.items():
                    if len(cand.history) >= PROMOTION_WINDOW and sum(cand.history) >= PROMOTION_THRESHOLD:
                        to_promote.append((tid, cand.bbox))

                if to_promote or to_remove:
                    survivor_bboxes = {oid: get_bbox_from_mask(m) for oid, m in masks_by_id.items() if oid not in to_remove}
                    sam2_predictor.load_first_frame(frame_rgb)

                    for oid in to_remove:
                        miss_counts.pop(oid, None)

                    for oid, bbox in survivor_bboxes.items():
                        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=oid, bbox=bbox_arr)
                    for temp_id, bbox in to_promote:
                        new_id = tracker.next_id
                        tracker.next_id += 1
                        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=new_id, bbox=bbox_arr)
                        miss_counts[new_id] = 0
                        del candidates[temp_id]

                for oid, mask in masks_by_id.items():
                    bbox = get_bbox_from_mask(mask)

                    cx_raw = (bbox[0] + bbox[2]) / 2
                    cy_raw = (bbox[1] + bbox[3]) / 2
                    bbox_max_dim = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

                    if oid not in crop_state:
                        crop_state[oid] = {
                            'cx': cx_raw,
                            'cy': cy_raw,
                            'crop_box_size': int(bbox_max_dim * CROP_SIZE_PADDING),
                            'cycle_start_frame': frame_count,
                            'frames': [],
                            'is_drowning': False,
                        }
                    else:
                        s = crop_state[oid]
                        if len(s['frames']) >= cycle_frames:
                            s['is_drowning'] = classify_clip(s['frames'], model, device)
                            s['frames'] = []
                            s['crop_box_size'] = int(bbox_max_dim * CROP_SIZE_PADDING)
                            s['cycle_start_frame'] = frame_count

                        s['cx'] = CROP_EMA_PADDING * cx_raw + (1 - CROP_EMA_PADDING) * s['cx']
                        s['cy'] = CROP_EMA_PADDING * cy_raw + (1 - CROP_EMA_PADDING) * s['cy']

                    crop = crop_with_padding(frame, crop_state[oid])
                    if crop is not None:
                        crop_state[oid]['frames'].append(crop)

                    is_drowning = crop_state[oid]['is_drowning']
                    color = (0, 0, 255) if is_drowning else (0, 255, 0)  # drowning=빨강, normal=초록 (BGR)
                    # color = (0, 0, 255) if is_drowning else color_for_id(oid)  # ← normal을 ID별 색으로 보려면 이 줄로 교체
                    label = f"Drowning ID:{oid}" if is_drowning else f"ID:{oid}"
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if roi_polygon is not None:
                cv2.polylines(frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

            fps = 1.0 / (time.time() - t_start + 1e-6)
            cv2.putText(frame, f"FPS : {fps:.1f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if out is None:
                out = create_video_writer(video_path, frame, fps=FPS)
            out.write(frame)

            if SHOW:
                cv2.imshow("test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if out is not None:
            out.release()
        if SHOW:
            cv2.destroyAllWindows()


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 모델은 한 번만 로드 후 모든 영상에 재사용
    yolo_model = YOLO(yolo_weight_path)
    model = load_lora_model(device)

    GlobalHydra.instance().clear()
    sam2_config_dir = os.path.join(_REPO_ROOT, 'sam2_streaming', 'sam2', 'configs')
    with initialize_config_dir(config_dir=sam2_config_dir, version_base=None):
        sam2_predictor = build_sam2_camera_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT)

    for video_path in VIDEO_PATHS:
        print(f"\n===== 처리 시작: {video_path} =====")
        run_demo(video_path, yolo_model, model, sam2_predictor, device)


if __name__ == "__main__":
    main()
