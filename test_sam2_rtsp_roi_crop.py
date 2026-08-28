import cv2
import numpy as np
import torch
import sys, os
import threading
import time
import glob
import json
from queue import Queue
from hydra import initialize_config_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_streaming'))

from sam2.build_sam import build_sam2_camera_predictor
from hydra.core.global_hydra import GlobalHydra
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
from ultralytics import YOLO


# ============================================================
# Path / Model 설정
# ============================================================
VIDEOS_DIR       = 'datasets/videos/'
VIDEO_EXTENSIONS = ['.mp4']
WEIGHT_PATH      = 'weights/yolo/swim_0627.pt'
SAM2_CHECKPOINT  = 'sam2/checkpoints/sam2.1_hiera_small.pt'
SAM2_MODEL_CFG   = 'sam2.1/sam2.1_hiera_s.yaml'

# ============================================================
# 추적/매칭 임계값
# ============================================================
DET_DET_IOU_THRESHOLD       = 0.3
MASK_OVERLAP_THRESHOLD      = 0.85
MASK_DET_IOU_THRESHOLD      = 0.15
MASK_DET_IOU_GATE_THRESHOLD = 0.15

PROMOTION_THRESHOLD         = 3
PROMOTION_WINDOW            = 5
MIN_MASK_AREA               = 60

# ============================================================
# 시간 기반 상수 — 영상 fps에 따라 frame 수 동적 계산
# ============================================================
LOST_TIMEOUT_SECONDS         = 3.0
CANDIDATE_MAX_UNSEEN_SECONDS = 10.0 / 30.0      # 30fps 기준 10 frame 동등
CROP_CYCLE_SECONDS           = 4

# ============================================================
# CROP 설정
# ============================================================
CROP_OUTPUT_DIR    = 'crops'
CROP_OUTPUT_SIZE   = 224
CROP_SIZE_PADDING  = 1.8
CROP_EMA_ALPHA     = 0.3


# ============================================================
# Video Reader (모든 프레임 보존, 큐 가득 차면 reader가 block)
# ============================================================
class VideoReader:
    def __init__(self, video_path, queue_size=32):
        self.video_path = video_path
        self.frame_queue = Queue(maxsize=queue_size)
        self.stop_flag = False
        self.thread = None

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
                self.frame_queue.put((ret, frame))   # 가득 차면 main 처리에 동기화
        finally:
            cap.release()
            self.frame_queue.put((False, None))

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stop_flag = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================
# Tracker / Candidate
# ============================================================
class TrackManager:
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 1


class Candidate:
    """SAM2 등록 전 객체 후보 (M-of-N 검증용)"""
    def __init__(self, bbox, frame_idx):
        self.bbox = bbox
        self.history = [True]
        self.last_seen = frame_idx


# ============================================================
# 기하 / 마스크 유틸
# ============================================================
def calculate_iou(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    return inter / (box1_area + box2_area - inter)


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


# ============================================================
# ROI 로드 (사전 저장된 json)
# ============================================================
def get_roi_path(video_path):
    return os.path.splitext(video_path)[0] + '.roi.json'


def load_roi(video_path):
    """xxx.roi.json 로드. 없거나 점 3개 미만이면 None."""
    roi_path = get_roi_path(video_path)
    if not os.path.exists(roi_path):
        return None
    with open(roi_path, 'r') as f:
        data = json.load(f)
    points = data.get('points', [])
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def is_bbox_in_roi(bbox, polygon):
    """bbox 중심이 polygon 내부에 있는지. polygon=None이면 항상 True."""
    if polygon is None or len(polygon) < 3:
        return True
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0


# ============================================================
# 중복 트랙 감지
# ============================================================
def find_duplicate_tracks(masks_by_id, yolo_bboxes, mask_overlap_threshold, mask_det_iou_threshold):
    yolo_matched = {}
    for oid, mask in masks_by_id.items():
        bbox = get_bbox_from_mask(mask)
        yolo_matched[oid] = any(
            calculate_iou(bbox, det) > mask_det_iou_threshold for det in yolo_bboxes
        )

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


# ============================================================
# CROP — 위치 EMA + 4초 주기 크기 갱신 + 정사각형 검정 패딩
# ============================================================
def open_clip_writer(video_name, obj_id, clip_index, fps,
                     output_dir=CROP_OUTPUT_DIR, size=(CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE)):
    """경로: crops/{video_name}/{video_name}_id_{id:04d}_clip_{clip:03d}.mp4"""
    obj_dir = os.path.join(output_dir, video_name)
    os.makedirs(obj_dir, exist_ok=True)
    file_name = f"{video_name}_id_{obj_id:04d}_clip_{clip_index:03d}.mp4"
    file_path = os.path.join(obj_dir, file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(file_path, fourcc, fps, size), file_path


def release_clip_writer(state, min_frames):
    """writer release. 클립이 min_frames 미만이면 mp4 파일 자동 삭제."""
    if state.get('writer') is None:
        return
    state['writer'].release()
    state['writer'] = None
    if state.get('clip_frame_count', 0) < min_frames:
        file_path = state.get('current_file_path')
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


def update_crop_state(crop_state, video_name, oid, bbox, frame_count, fps, cycle_frames):
    """obj_id별 crop 상태 갱신 + 클립 writer 관리."""
    cx_raw = (bbox[0] + bbox[2]) / 2
    cy_raw = (bbox[1] + bbox[3]) / 2
    bbox_max_dim = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    if oid not in crop_state:
        # 첫 등장 — 크기 결정 + 첫 클립 writer 생성
        writer, file_path = open_clip_writer(video_name, oid, 0, fps)
        crop_state[oid] = {
            'cx': cx_raw,
            'cy': cy_raw,
            'locked_size': int(bbox_max_dim * CROP_SIZE_PADDING),
            'cycle_start_frame': frame_count,
            'clip_index': 0,
            'writer': writer,
            'current_file_path': file_path,
            'clip_frame_count': 0,
        }
        return crop_state[oid]

    s = crop_state[oid]

    # 4초 채운 정상 클립 → release (min=0이면 무조건 keep) + 새 클립 시작
    if frame_count - s['cycle_start_frame'] >= cycle_frames:
        release_clip_writer(s, min_frames=0)   # 4초 정확히 채웠으니 keep
        s['locked_size'] = int(bbox_max_dim * CROP_SIZE_PADDING)
        s['cycle_start_frame'] = frame_count
        s['clip_index'] += 1
        writer, file_path = open_clip_writer(video_name, oid, s['clip_index'], fps)
        s['writer'] = writer
        s['current_file_path'] = file_path
        s['clip_frame_count'] = 0

    # 위치 EMA
    s['cx'] = CROP_EMA_ALPHA * cx_raw + (1 - CROP_EMA_ALPHA) * s['cx']
    s['cy'] = CROP_EMA_ALPHA * cy_raw + (1 - CROP_EMA_ALPHA) * s['cy']
    return s


def crop_with_padding(frame, smooth_state, output_size=CROP_OUTPUT_SIZE):
    """smooth_state 기준 정사각형 crop. 영상 밖은 검정 패딩 + output_size로 resize."""
    cx = int(smooth_state['cx'])
    cy = int(smooth_state['cy'])
    size = smooth_state['locked_size']
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


def close_all_clip_writers(crop_state, min_frames):
    """모든 obj writer release + 짧은 클립 자동 삭제 (영상 종료 시 호출)."""
    for oid, s in list(crop_state.items()):
        release_clip_writer(s, min_frames)


# ============================================================
# 한 영상 처리
# ============================================================
def process_one_video(video_path, sam2_predictor, yolo_model):
    """한 영상 처리. SAM2/YOLO는 외부에서 받음. 영상별 상태는 함수 안에 격리."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # ROI 로드
    polygon = load_roi(video_path)
    if polygon is None:
        print(f"  [skip] ROI 없음 ({get_roi_path(video_path)})")
        return

    # fps probe → 시간 기반 frame 수 계산
    cap_probe = cv2.VideoCapture(video_path)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    cap_probe.release()
    lost_timeout         = int(round(fps * LOST_TIMEOUT_SECONDS))
    candidate_max_unseen = int(round(fps * CANDIDATE_MAX_UNSEEN_SECONDS))
    crop_cycle_frames    = int(round(fps * CROP_CYCLE_SECONDS))
    print(f"  fps={fps:.2f}  lost_timeout={lost_timeout}f  "
          f"candidate_max_unseen={candidate_max_unseen}f  crop_cycle={crop_cycle_frames}f")

    # 영상별 상태 초기화
    tracker          = TrackManager()
    miss_counts      = {}
    candidates       = {}
    crop_state       = {}
    next_temp_id     = 1
    frame_count      = 0
    sam2_initialized = False

    with VideoReader(video_path) as video_reader:
        time.sleep(0.3)

        while True:
            ret, frame = video_reader.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # ── SAM2 init: ROI 안 첫 detection 시점에 init (첫 프레임이 비어 있어도 다음 프레임 계속 시도) ──
            if not sam2_initialized:
                results = yolo_model.predict(frame, verbose=False)
                valid_bboxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    if is_bbox_in_roi([x1, y1, x2, y2], polygon):
                        valid_bboxes.append((x1, y1, x2, y2))
                if not valid_bboxes:
                    continue
                sam2_predictor.load_first_frame(frame_rgb)
                for x1, y1, x2, y2 in valid_bboxes:
                    bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tracker.next_id, bbox=bbox_arr)
                    tracker.next_id += 1
                sam2_initialized = True
                print(f"  [init] SAM2 시작 @ frame {frame_count}, 객체 {len(valid_bboxes)}개")
                continue

            # ── SAM2 추적 ──
            obj_ids, mask_logits = sam2_predictor.track(frame_rgb)
            masks_all = (mask_logits > 0.0).squeeze(1).cpu().numpy()
            masks_by_id = {
                oid: masks_all[i] for i, oid in enumerate(obj_ids)
                if masks_all[i].sum() >= MIN_MASK_AREA
            }

            # ── YOLO 탐지 (ROI 안만) ──
            results = yolo_model.predict(frame, verbose=False)
            yolo_bboxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                if not is_bbox_in_roi([x1, y1, x2, y2], polygon):
                    continue
                yolo_bboxes.append([x1, y1, x2, y2])

            # ── 사람 검증 (miss_counts) ──
            for oid in obj_ids:
                if oid in masks_by_id:
                    obj_bbox = get_bbox_from_mask(masks_by_id[oid])
                    matched_yolo = any(
                        calculate_iou(obj_bbox, yb) > MASK_DET_IOU_THRESHOLD
                        for yb in yolo_bboxes
                    )
                    miss_counts[oid] = 0 if matched_yolo else miss_counts.get(oid, 0) + 1
                else:
                    miss_counts[oid] = miss_counts.get(oid, 0) + 1

            # ── 삭제 대상 결정 ──
            to_remove = {oid for oid, miss in miss_counts.items() if miss > lost_timeout}
            to_remove |= find_duplicate_tracks(
                masks_by_id, yolo_bboxes, MASK_OVERLAP_THRESHOLD, MASK_DET_IOU_THRESHOLD
            )
            for oid, mask in masks_by_id.items():
                if not is_bbox_in_roi(get_bbox_from_mask(mask), polygon):
                    to_remove.add(oid)

            # ── 신규 후보 풀 갱신 ──
            matched_candidates_ids = set()
            for det_bbox in yolo_bboxes:
                is_existing = any(
                    calculate_iou(det_bbox, get_bbox_from_mask(m)) > MASK_DET_IOU_GATE_THRESHOLD
                    for m in masks_by_id.values()
                )
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
            candidates = {
                tid: cand for tid, cand in candidates.items()
                if frame_count - cand.last_seen <= candidate_max_unseen
            }

            # ── 승격 자격 후보 ──
            to_promote = [
                (tid, cand.bbox)
                for tid, cand in candidates.items()
                if len(cand.history) >= PROMOTION_WINDOW and sum(cand.history) >= PROMOTION_THRESHOLD
            ]

            # ── SAM2 재등록 (_full_resync) ──
            if to_promote or to_remove:
                survivor_bboxes = {
                    oid: get_bbox_from_mask(m)
                    for oid, m in masks_by_id.items()
                    if oid not in to_remove
                }

                # 삭제 obj 정리 (writer release + 짧은 클립 자동 삭제) — 항상 수행
                for oid in to_remove:
                    miss_counts.pop(oid, None)
                    if oid in crop_state:
                        release_clip_writer(crop_state[oid], min_frames=crop_cycle_frames)
                        crop_state.pop(oid, None)

                total_prompts = len(survivor_bboxes) + len(to_promote)
                if total_prompts == 0:
                    # 빈 상태로 SAM2 init 시 다음 track()이 assert 실패 → init 모드로 복귀
                    sam2_initialized = False
                    print(f"  [reinit] frame {frame_count}: 모든 obj 제거됨, 다음 detection 대기")
                else:
                    sam2_predictor.load_first_frame(frame_rgb)
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

            # ── CROP 저장 (4초 mp4 단위) ──
            for oid, mask in masks_by_id.items():
                if oid in to_remove:
                    continue
                bbox = get_bbox_from_mask(mask)
                state = update_crop_state(crop_state, video_name, oid, bbox, frame_count,
                                          fps, crop_cycle_frames)
                crop_img = crop_with_padding(frame, state)
                if crop_img is not None and state.get('writer') is not None:
                    state['writer'].write(crop_img)
                    state['clip_frame_count'] += 1

            # 진행 표시 (1초마다)
            if frame_count % 30 == 0:
                print(f"  frame {frame_count}", end='\r')

        # 영상 종료 — 모든 writer 정리 + 짧은 클립 삭제
        close_all_clip_writers(crop_state, min_frames=crop_cycle_frames)
        print(f"  [done] frame_count={frame_count}")


# ============================================================
# Main — 폴더 내 모든 영상 일괄 처리
# ============================================================
def find_videos(videos_dir):
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(videos_dir, f'*{ext}')))
    return sorted(files)


def main():
    if not torch.cuda.is_available():
        print("CUDA 필요")
        return

    # 모델 1회 로드 (영상 간 재사용)
    print("YOLO 로드 중...")
    yolo_model = YOLO(WEIGHT_PATH)

    print("SAM2 로드 중...")
    GlobalHydra.instance().clear()
    sam2_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'sam2_streaming', 'sam2', 'configs')
    with initialize_config_dir(config_dir=sam2_config_dir, version_base=None):
        sam2_predictor = build_sam2_camera_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT)

    # 영상 폴더 처리
    videos = find_videos(VIDEOS_DIR)
    if not videos:
        print(f"영상 없음: {VIDEOS_DIR}")
        return

    print(f"\n총 {len(videos)}개 영상 처리 시작\n")

    for i, video_path in enumerate(videos, 1):
        name = os.path.basename(video_path)
        print(f"\n[{i}/{len(videos)}] {name}")
        try:
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                process_one_video(video_path, sam2_predictor, yolo_model)
        except Exception as e:
            print(f"  [error] {e}")
            import traceback
            traceback.print_exc()

    print("\n=== 모든 영상 처리 완료 ===")


if __name__ == "__main__":
    main()
