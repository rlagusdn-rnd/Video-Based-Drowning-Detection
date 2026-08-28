import cv2
import numpy as np
import torch
import sys, os
import threading
import time
from queue import Queue
from hydra import initialize_config_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_streaming'))

from sam2.build_sam import build_sam2_camera_predictor

from hydra.core.global_hydra import GlobalHydra
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

from ultralytics import YOLO

VIDEO_PATH       = 'datasets/videos/video_submarine072_clip.mp4'
yolo_weight_path     = 'weights/yolo/swim_0627.pt'
SAM2_CHECKPOINT  = 'sam2/checkpoints/sam2.1_hiera_small.pt'
# SAM2_CHECKPOINT  = 'sam2/checkpoints/sam2.1_hiera_base_plus.pt'
SAM2_MODEL_CFG   = 'sam2.1/sam2.1_hiera_s.yaml'
# SAM2_MODEL_CFG   = 'sam2.1/sam2.1_hiera_b+.yaml'

DET_DET_IOU_THRESHOLD = 0.3         # 이전/현재 프레임 DET 결과 비교 [기존 객체 후보와 신규 yolo 검출 결과 매칭에 사용]
MASK_OVERLAP_THRESHOLD = 0.8        # SAM2 마스크 중복 판단 임계값 [중복 추적 감지에 사용]
MASK_DET_IOU_THRESHOLD = 0.15       # SAM2 MASK BBOX / YOLO DET 결과 비교 [추적 객체의 사람 판단 / 신규 객체 후보 등록 시 기존 객체와의 매칭 판단에 사용]
MASK_DET_IOU_GATE_THRESHOLD = 0.15  # SAM2 MASK BBOX / YOLO DET 결과 비교 [신규 객체 후보 등록 시 기존 SAM2 객체와의 매칭 판단에 사용]

PROMOTION_THRESHOLD = 3             # 새 객체로 승격하기 위한 최소 매칭 프레임 수 (PROMOTION_WINDOW 내에서 True인 프레임 수)
PROMOTION_WINDOW = 5                # 객체 승격 후보 검증 프레임 수
LOST_TIMEOUT = 45                   # 객체를 잃었다고 판단하기 위한 프레임 수 (30fps * 3초)
CANDIDATE_MAX_UNSEEN_FRAMES = 10    # 객체 승격 후보가 너무 오래 보이지 않는 경우 제거하기 위한 프레임 수
MIN_MASK_AREA = 60                  # SAM2 마스크 최소 면적 기준(픽셀 값)

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
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put((ret, frame))
                time.sleep(0.033)

        finally:
            cap.release()
            # 종료 신호 전송
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
    """
    SAM2에 등록하기 전 객체 후보 정보 저장 클래스
    bbox : 가장 최근 매칭된 yolo bbox
    history : 최근 PROMOTION_WINDOW개 프레임에서 매칭 여부 기록 (True/False 리스트)
    last_seen : 마지막으로 매칭된 프레임 번호
    """
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

def get_bbox_from_mask(mask):
    """
    : 한명의 객체에 대한 segmentation 결과(mask)를 bounding box로 변환하는 함수
    입력 : mask (boolean으로 되어있는 1080 x 1920 array)
    출력 : bbox (bounding box 좌표)
    """
    rows = np.any(mask, axis=1) # true가 있는 행들
    cols = np.any(mask, axis=0) # true가 있는 열들
    y1, y2 = np.where(rows)[0][[0, -1]] # [0]은 처음, [-1]은 마지막 / np.where(rows)[0] ture인 행번호들 ex : [3, 4, 7]
    x1, x2 = np.where(cols)[0][[0, -1]] # [0]은 처음, [-1]은 마지막
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    return bbox

def color_for_id(obj_id):
      """obj_id로부터 결정적인 BGR 색깔을 만든다 (시각화용)."""
      hue = (obj_id * 47) % 180
      bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
      return int(bgr[0]), int(bgr[1]), int(bgr[2])

def find_duplicate_tracks(masks_by_id: dict, yolo_bboxes: list, mask_overlap_threshold: float, mask_det_iou_threshold: float) -> set:
    """mask overlap + YOLO 매칭으로 중복 트랙 감지
    YOLO에 사람으로 잡힌 obj는 살리고, 안 잡힌 쪽을 우선 삭제
    둘 다 사람이면 작은 쪽 삭제

      return : to_remove_local (중복으로 판단되어 제거할 객체 ID 집합)
    """
    yolo_matched = {}
    for oid, mask in masks_by_id.items():
        bbox = get_bbox_from_mask(mask)
        yolo_matched[oid] = any(calculate_iou(bbox, det) > mask_det_iou_threshold for det in yolo_bboxes) # SAM2 MASK bbox와 yolo bbox 간의 iou 기준으로 매칭 여부 판단 (True/False)

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
            if mask_overlap <= mask_overlap_threshold: # MASK 간 중복 판정
                continue

            # ── 결정: YOLO 매칭 우선 ──
            a_is_person = yolo_matched[a]
            b_is_person = yolo_matched[b]

            if a_is_person and not b_is_person:
                to_remove_local.add(b)              # b가 부유물
            elif b_is_person and not a_is_person:
                to_remove_local.add(a)              # a가 부유물
            else:
                # 둘 다 사람 / 둘 다 사람 아님 → 작은 쪽 삭제 (기존 정책)
                loser = a if area_a < area_b else b
                to_remove_local.add(loser)
    return to_remove_local

@torch.inference_mode()
@torch.amp.autocast('cuda')
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if not os.path.isfile(VIDEO_PATH):
        print(f"there is no video file at {VIDEO_PATH}")
        return
    # yolo 모델로드
    yolo_model = YOLO(yolo_weight_path)

    # sam2 모델로드
    GlobalHydra.instance().clear()
    sam2_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_streaming', 'sam2', 'configs')
    with initialize_config_dir(config_dir=sam2_config_dir, version_base=None):
        sam2_predictor = build_sam2_camera_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT)

    tracker = TrackManager()
    frame_count = 0
    yolo_bboxes = []
    miss_counts = {}                # SAM2가 추적 중인 객체가 현재 프레임에서 매칭되지 않은 횟수 관리
    candidates = {}
    next_temp_id = 1

    with VideoReader(VIDEO_PATH) as video_reader:
        time.sleep(0.5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30, (1920, 1080))

        while True:
            t_start = time.time()
            ret, frame = video_reader.read()
            if not ret:
                print(f"영상 종료: frame_count={frame_count}")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # 첫 프레임 처리: YOLO로 탐지된 bbox를 SAM2에 등록
            if frame_count == 1:
                sam2_predictor.load_first_frame(frame_rgb)
                results = yolo_model.predict(frame, verbose=False)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = np.array([[x1, y1], [x2, y2]], dtype=np.float32) # SAM2 입력 형식으로 변환
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=tracker.next_id, bbox=bbox) # SAM2에 첫 프레임과 bbox로 객체 등록
                    tracker.next_id += 1

            # 이후 프레임 처리: SAM2로 객체 추적, YOLO로 신규 객체 탐지 및 후보 관리
            else:
                obj_ids, mask_logits = sam2_predictor.track(frame_rgb) # SAM2로 객체 추적 (출력 : [1, 2]객체 ID 리스트, (N, 1, H, W) (obj수, 채널, 높이, 너비))
                masks_by_id = {}

                masks_all = (mask_logits > 0.0).squeeze(1).cpu().numpy()
                for i, oid in enumerate(obj_ids):
                    if masks_all[i].sum() >= MIN_MASK_AREA: # 마스크가 충분히 큰 경우에만 bbox 업데이트
                        masks_by_id[oid] = masks_all[i]

                results = yolo_model.predict(frame, verbose=False)
                yolo_bboxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    yolo_bboxes.append([x1, y1, x2, y2])

                # miss_counts 관리
                for oid in obj_ids:
                    if oid in masks_by_id:
                        obj_bbox = get_bbox_from_mask(masks_by_id[oid])
                        matched_yolo = any(calculate_iou(obj_bbox, yolo_bbox) > MASK_DET_IOU_THRESHOLD for yolo_bbox in yolo_bboxes) # SAM2 MASK와 yolo bbox 간의 iou 기준으로 매칭 여부 판단 (True/False)
                        if matched_yolo:
                            miss_counts[oid] = 0  # miss_counts 초기화
                        else:
                            miss_counts[oid] = miss_counts.get(oid, 0) + 1 # miss_counts 누적
                    else:
                        miss_counts[oid] = miss_counts.get(oid, 0) + 1 # SAM2가 추적 중인 객체가 현재 프레임에서 매칭되지 않은 경우 누적
                to_remove = {oid for oid, miss in miss_counts.items() if miss > LOST_TIMEOUT} # LOST_TIMEOUT 초과한 객체 ID 수집

                # 마스크 중복 검사
                to_remove |= find_duplicate_tracks(masks_by_id, yolo_bboxes, MASK_OVERLAP_THRESHOLD, MASK_DET_IOU_THRESHOLD) # SAM2 MASK 중복 검사

                matched_candidates_ids = set()
                for det_bbox in yolo_bboxes:
                    is_existing = any(calculate_iou(det_bbox, get_bbox_from_mask(m)) > MASK_DET_IOU_GATE_THRESHOLD for m in masks_by_id.values()) # SAM2 MASK와 YOLO BBOX간 iou 기준으로 매칭 여부 판단 (True/False)
                    if is_existing:
                        continue

                    # 기존 객체 후보들과 yolo 검출 결과 매칭 (매칭된 후보는 history에 True 추가, bbox 업데이트, last_seen 업데이트)
                    matched = False
                    for tid, cand in candidates.items(): # 첫 진입 시 candidates가 비어 있으므로 패스, 이후엔 누적된 후보와 매칭 시도
                        if calculate_iou(det_bbox, cand.bbox) > DET_DET_IOU_THRESHOLD: # yolo bbox와 후보 bbox 간의 iou 기준으로 매칭 여부 판단 (True/False)
                            cand.history.append(True)
                            cand.bbox = det_bbox
                            cand.last_seen = frame_count
                            matched_candidates_ids.add(tid)
                            matched = True
                            break

                    # 매칭된 후보가 없으면 새로운 후보로 등록
                    if not matched:
                        candidates[next_temp_id] = Candidate(det_bbox, frame_count)
                        matched_candidates_ids.add(next_temp_id)
                        next_temp_id += 1

                # 매칭되지 않은 후보들은 history에 False 추가
                for tid, cand in candidates.items():
                    if tid not in matched_candidates_ids:
                        cand.history.append(False)

                # candidate의 history 값 길이 유지
                for cand in candidates.values():
                    if len(cand.history) > PROMOTION_WINDOW:
                        cand.history = cand.history[-PROMOTION_WINDOW:]

                # unseen frame동안 매칭되지 않았던 후보 제거하고 덮어쓰기 (안보인 객체 제거)
                candidates = {tid: cand for tid, cand in candidates.items() if frame_count - cand.last_seen <= CANDIDATE_MAX_UNSEEN_FRAMES}

                to_promote = []
                # history 기준으로 SAM2에 등록할 후보 선정 (history 길이 >= PROMOTION_WINDOW, True 개수 >= PROMOTION_THRESHOLD)
                for tid, cand in candidates.items():
                    if len(cand.history) >= PROMOTION_WINDOW and sum(cand.history) >= PROMOTION_THRESHOLD:
                        to_promote.append((tid, cand.bbox))

                # SAM2에 mask 등록 (신규 객체 포함/놓침 객체 제외)
                if to_promote or to_remove:
                    # SAM2가 아직 추적 중인 (제거 대상에 포함 안된)객체들의 bbox 저장 (masks_by_id -> survivor_bboxes)
                    survivor_bboxes = {oid: get_bbox_from_mask(m) for oid, m in masks_by_id.items() if oid not in to_remove}
                    sam2_predictor.load_first_frame(frame_rgb) # SAM2 재등록

                    for oid in to_remove:
                        miss_counts.pop(oid, None) # miss_counts에서 제거

                    # SAM2에 기존 생존 객체 등록
                    for oid, bbox in survivor_bboxes.items():
                        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=oid, bbox=bbox_arr)
                    # SAM2에 신규 객체 등록
                    for temp_id, bbox in to_promote:
                        new_id = tracker.next_id
                        tracker.next_id += 1
                        bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                        sam2_predictor.add_new_prompt(frame_idx=0, obj_id=new_id, bbox=bbox_arr)
                        miss_counts[new_id] = 0 # miss_counts 초기화
                        del candidates[temp_id] # 후보에서 승격된 객체 제거

                for oid, mask in masks_by_id.items():
                    color = color_for_id(oid)
                    frame[mask] = (frame[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    bbox = get_bbox_from_mask(mask)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, f"ID:{oid}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            fps = 1.0 / (time.time() - t_start + 1e-6)
            cv2.putText(frame, f"FPS : {fps:.1f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            cv2.imshow("test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    # 해결해야 할 부분. bbox 크기 변동 심함 -> track bbox의 중앙 좌표 기준으로 bbox 생성. bbox의 크기는 이전 프레임과 smooth 비교
