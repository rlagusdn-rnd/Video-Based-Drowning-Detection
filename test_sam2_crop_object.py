"""
test_sam2_crop_object.py
========================
test_sam2_rtsp_roi_crop.py 가 못 잡는 익수 대상자를 수동 선택해서 클립화하는 도구.

흐름:
  1) 영상 폴더에서 차례대로 영상을 열고, 사용자가 SPACE/BS 로 시작 프레임 탐색
  2) 익수 대상자를 좌클릭 → ENTER 로 첫 4초 클립의 anchor 등록
  3) 4초 동안 SAM2 단독 추적 (YOLO 검증/개입 없음, 그냥 진행) → 클립 저장
  4) 4초 도달 시점에 자동 일시정지 + 사용자 anchor UI → 다음 4초 anchor 갱신
  5) 반복

핵심: 매 클립이 사용자 anchor 로 시작 → 4초 통째로 채움 → 무조건 저장.
      클립 안 framing 흔들림 방지를 위해 cx/cy/locked_size 모두 anchor 시점 값으로
      고정 (EMA 안 함).

UX (시작 프레임 선택):
  SPACE/D     +1초          BS/A        -1초
  . / >       +1프레임      , / <       -1프레임
  '           +10초         ;           -10초
  ]           +1분          [           -1분
  =           +10분         -           -10분
  좌클릭      positive      우클릭      negative
  U           마지막 점 undo R           점 전부 리셋
  ENTER       확정 (추적 시작)
  N           이 영상 스킵   Q/ESC       전체 종료

UX (트래킹 중 — Live 화면):
  SPACE/P     일시정지 토글 (검토용)
  N           이 영상만 스킵 (current partial 폐기, 다음 영상으로)
  Q/ESC       전체 종료

UX (Anchor UI — 매 4초마다 자동 호출):
  좌/우클릭   pos/neg point  U / R       undo / reset
  ENTER       새 mask 로 다음 4초 anchor 갱신
  C           현재 SAM2 mask 의 bbox 로 anchor (클릭 없이 — 추적 잘 될 때)
  N           영상 종료 (current partial 폐기)
  E           강제 anchor UI 비활성화 (이후 SAM2 자체로 진행)
  Q/ESC       전체 종료
"""
import cv2
import numpy as np
import torch
import sys
import os
import glob
import json
from hydra import initialize_config_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_streaming'))

from sam2.build_sam import build_sam2_camera_predictor
from hydra.core.global_hydra import GlobalHydra
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
from ultralytics import YOLO


# ============================================================
# Path / Model
# ============================================================
VIDEOS_DIR       = 'datasets/videos/'
VIDEO_EXTENSIONS = ['.mp4']
WEIGHT_PATH      = 'weights/yolo/swim_0627.pt'
SAM2_CHECKPOINT  = 'sam2/checkpoints/sam2.1_hiera_small.pt'
SAM2_MODEL_CFG   = 'sam2.1/sam2.1_hiera_s.yaml'

# ============================================================
# 추적/매칭
# ============================================================
MIN_MASK_AREA      = 60          # 이 이하 mask 는 무효
USER_OBJ_ID        = 1           # 사용자가 선택한 단일 객체 ID

# ============================================================
# CROP
# ============================================================
CROP_OUTPUT_DIR    = 'crop_object'
CROP_OUTPUT_SIZE   = 224
CROP_SIZE_PADDING  = 1.8
CROP_CYCLE_SECONDS = 4
CROP_CYCLE_FRAMES  = 30 * CROP_CYCLE_SECONDS    # 120 프레임 = 4초 (30fps 기준)
CROP_MIN_FRAMES_END = CROP_CYCLE_FRAMES         # 영상 끝 partial 클립은 미만이면 삭제
CROP_EMA_ALPHA     = 0.3                        # cx/cy 객체 추적 smoothing (작을수록 부드러움)

# ============================================================
# Selection UI hint
# ============================================================
SELECT_HINT = ("[1f:,/.  1s:BS/SPACE  10s:;/'  1m:[/]  10m:-/=  "
               "L:pos R:neg ENTER:ok U:undo R:reset N:skip Q:quit]")


# ============================================================
# 기하 / 마스크 유틸
# ============================================================
def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


# ============================================================
# ROI
# ============================================================
def get_roi_path(video_path):
    return os.path.splitext(video_path)[0] + '.roi.json'


def load_roi(video_path):
    roi_path = get_roi_path(video_path)
    if not os.path.exists(roi_path):
        return None
    with open(roi_path, 'r') as f:
        data = json.load(f)
    points = data.get('points', [])
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def is_point_in_roi(x, y, polygon):
    if polygon is None or len(polygon) < 3:
        return True
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0


def is_bbox_in_roi(bbox, polygon):
    if polygon is None or len(polygon) < 3:
        return True
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0


# ============================================================
# 공통 — point 기반 mask 새로고침
# ============================================================
def _refresh_mask(sam2_predictor, points, labels, obj_id):
    if len(points) == 0:
        return None
    pts = np.array(points, dtype=np.float32)
    lbs = np.array(labels, dtype=np.int32)
    _, _, mask_logits = sam2_predictor.add_new_prompt(
        frame_idx=0, obj_id=obj_id,
        points=pts, labels=lbs, clear_old_points=True,
    )
    return (mask_logits[0] > 0.0).squeeze(0).cpu().numpy()


# ============================================================
# 시작 프레임 + 객체 선택 UI
# ============================================================
def _draw_selection_overlay(base_bgr, polygon, points, labels, mask, info_text=""):
    canvas = base_bgr.copy()
    if polygon is not None and len(polygon) >= 3:
        cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2)
    if mask is not None and mask.any():
        color_layer = np.zeros_like(canvas)
        color_layer[mask] = (255, 0, 255)
        canvas = cv2.addWeighted(canvas, 1.0, color_layer, 0.45, 0)
        bbox = get_bbox_from_mask(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)
    for (x, y), lab in zip(points, labels):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(canvas, (int(x), int(y)), 7, color, -1)
        cv2.circle(canvas, (int(x), int(y)), 9, (255, 255, 255), 2)
    pos = sum(labels)
    txt = f"{info_text}   points: {len(points)} (pos={pos}, neg={len(labels)-pos})"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, txt, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def select_object(video_path, polygon, sam2_predictor):
    """
    return:
      ('go',   start_idx, frame_bgr, mask, cap)
      ('skip', None, None, None, None)
      ('quit', None, None, None, None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ('skip', None, None, None, None)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step      = max(1, int(round(fps)))         # 1 second
    step_10s  = step * 10
    step_1m   = step * 30
    step_10m  = step * 300

    window_name = f"{os.path.basename(video_path)}  {SELECT_HINT}"

    state = {
        'idx': -1,
        'frame_bgr': None,
        'points': [],
        'labels': [],
        'mask': None,
        'dirty': False,
    }

    def goto(target):
        target = max(0, min(total - 1, int(target)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, fr = cap.read()
        if not ok or fr is None:
            return False
        state['idx'] = target
        state['frame_bgr'] = fr
        state['points'].clear()
        state['labels'].clear()
        state['mask'] = None
        state['dirty'] = False
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        sam2_predictor.load_first_frame(rgb)
        sam2_predictor.frame_idx = 0   # 영상 간 누적 방지 (memory attention 품질 유지)
        return True

    if not goto(0):
        cap.release()
        return ('skip', None, None, None, None)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not is_point_in_roi(x, y, polygon):
                print(f"  [warn] positive 점 ({x},{y}) 가 ROI 밖 — 무시")
                return
            state['points'].append((x, y))
            state['labels'].append(1)
            state['dirty'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            state['points'].append((x, y))
            state['labels'].append(0)
            state['dirty'] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    def reload_prompt_after_edit():
        rgb = cv2.cvtColor(state['frame_bgr'], cv2.COLOR_BGR2RGB)
        sam2_predictor.load_first_frame(rgb)
        state['mask'] = None
        state['dirty'] = bool(state['points'])

    while True:
        if state['dirty']:
            state['mask'] = _refresh_mask(
                sam2_predictor, state['points'], state['labels'], USER_OBJ_ID
            )
            state['dirty'] = False

        info = f"frame {state['idx']}/{total-1}  t={state['idx']/fps:.2f}s"
        canvas = _draw_selection_overlay(
            state['frame_bgr'], polygon,
            state['points'], state['labels'], state['mask'], info
        )
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue

        if key in (32, ord('d'), ord('D')):
            goto(state['idx'] + step)
            continue
        if key in (8, ord('a'), ord('A')):
            goto(state['idx'] - step)
            continue

        if key in (ord('.'), ord('>')):
            goto(state['idx'] + 1)
            continue
        if key in (ord(','), ord('<')):
            goto(state['idx'] - 1)
            continue

        # ±10초
        if key == ord("'"):
            goto(state['idx'] + step_10s)
            continue
        if key == ord(';'):
            goto(state['idx'] - step_10s)
            continue

        # ±1분
        if key == ord(']'):
            goto(state['idx'] + step_1m)
            continue
        if key == ord('['):
            goto(state['idx'] - step_1m)
            continue

        # ±10분
        if key == ord('='):
            goto(state['idx'] + step_10m)
            continue
        if key == ord('-'):
            goto(state['idx'] - step_10m)
            continue

        if key in (13, 10):
            if state['mask'] is None or not state['mask'].any():
                print("  [warn] 마스크 없음 — 점을 1개 이상 찍어주세요.")
                continue
            if sum(state['labels']) == 0:
                print("  [warn] positive 점이 없습니다.")
                continue
            cv2.destroyWindow(window_name)
            return ('go', state['idx'], state['frame_bgr'], state['mask'], cap)

        if key in (ord('r'), ord('R')):
            state['points'].clear()
            state['labels'].clear()
            reload_prompt_after_edit()
            continue

        if key in (ord('u'), ord('U')):
            if state['points']:
                state['points'].pop()
                state['labels'].pop()
                reload_prompt_after_edit()
            continue

        if key in (ord('n'), ord('N')):
            cap.release()
            cv2.destroyWindow(window_name)
            return ('skip', None, None, None, None)

        if key in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyWindow(window_name)
            return ('quit', None, None, None, None)


# ============================================================
# Anchor UI — 매 4초마다 자동 호출
# ============================================================
def anchor_ui(frame_bgr, polygon, sam2_predictor,
              target_oid, last_known_bbox, status_text):
    """
    return: ('recover', new_bbox) | ('drop', None) | ('continue', None)
            | ('extend', None) | ('quit', None)

    내부에서 sam2_predictor.load_first_frame() 호출 → 호출자는 어떤 결과든
    이번 프레임 기준으로 SAM2 를 새 anchor 로 재등록해야 함.
    """
    window_name = (f"[ANCHOR id={target_oid}]  "
                   "L:pos R:neg ENTER:newAnchor C:keepBbox E:extend N:drop U:undo R:reset Q:quit")

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sam2_predictor.load_first_frame(rgb)
    sam2_predictor.frame_idx = 0

    state = {'points': [], 'labels': [], 'mask': None, 'dirty': False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not is_point_in_roi(x, y, polygon):
                print(f"  [warn] positive 점 ({x},{y}) 가 ROI 밖 — 무시")
                return
            state['points'].append((x, y))
            state['labels'].append(1)
            state['dirty'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            state['points'].append((x, y))
            state['labels'].append(0)
            state['dirty'] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    def reload_prompt_after_edit():
        sam2_predictor.load_first_frame(rgb)
        sam2_predictor.frame_idx = 0
        state['mask'] = None
        state['dirty'] = bool(state['points'])

    while True:
        if state['dirty']:
            state['mask'] = _refresh_mask(
                sam2_predictor, state['points'], state['labels'], target_oid
            )
            state['dirty'] = False

        canvas = frame_bgr.copy()

        if polygon is not None and len(polygon) >= 3:
            cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2)

        # 직전 알려진 객체 위치 (참고용 — 노랑 박스)
        if last_known_bbox is not None:
            x1, y1, x2, y2 = last_known_bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(canvas, f"prev id{target_oid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 새 mask 미리보기 (magenta)
        if state['mask'] is not None and state['mask'].any():
            color_layer = np.zeros_like(canvas)
            color_layer[state['mask']] = (255, 0, 255)
            canvas = cv2.addWeighted(canvas, 1.0, color_layer, 0.45, 0)
            new_bbox = get_bbox_from_mask(state['mask'])
            if new_bbox is not None:
                x1, y1, x2, y2 = new_bbox
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)

        for (x, y), lab in zip(state['points'], state['labels']):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(canvas, (int(x), int(y)), 7, color, -1)
            cv2.circle(canvas, (int(x), int(y)), 9, (255, 255, 255), 2)

        pos = sum(state['labels'])
        info = (f"{status_text}  points={len(state['points'])} "
                f"(pos={pos}, neg={len(state['labels'])-pos})")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(canvas, info, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue

        if key in (13, 10):
            if state['mask'] is None or not state['mask'].any():
                print("  [warn] 마스크 없음 — 점을 1개 이상 찍어주세요.")
                continue
            if sum(state['labels']) == 0:
                print("  [warn] positive 점이 없습니다.")
                continue
            new_bbox = get_bbox_from_mask(state['mask'])
            cv2.destroyWindow(window_name)
            return ('recover', new_bbox)

        if key in (ord('n'), ord('N')):
            cv2.destroyWindow(window_name)
            return ('drop', None)

        if key in (ord('c'), ord('C')):
            cv2.destroyWindow(window_name)
            return ('continue', None)

        if key in (ord('e'), ord('E')):
            cv2.destroyWindow(window_name)
            return ('extend', None)

        if key in (ord('q'), ord('Q'), 27):
            cv2.destroyWindow(window_name)
            return ('quit', None)

        if key in (ord('u'), ord('U')):
            if state['points']:
                state['points'].pop()
                state['labels'].pop()
                reload_prompt_after_edit()
            continue

        if key in (ord('r'), ord('R')):
            state['points'].clear()
            state['labels'].clear()
            reload_prompt_after_edit()
            continue


# ============================================================
# CROP — anchor 시점 cx/cy/locked_size 로 정사각형 검정 패딩
# ============================================================
def crop_with_padding(frame, cx, cy, size, output_size=CROP_OUTPUT_SIZE):
    cx = int(cx)
    cy = int(cy)
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


def open_clip_writer(video_name, obj_id, clip_index, output_dir=CROP_OUTPUT_DIR,
                     fps=30, size=(CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE)):
    obj_dir = os.path.join(output_dir, video_name)
    os.makedirs(obj_dir, exist_ok=True)
    file_name = f"{video_name}_id_{obj_id:04d}_clip_{clip_index:03d}.mp4"
    file_path = os.path.join(obj_dir, file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(file_path, fourcc, fps, size), file_path


# ============================================================
# resync helper — load_first_frame + 단일 객체 다시 add_new_prompt
# ============================================================
def full_resync_single(sam2_predictor, frame_rgb, oid, bbox):
    sam2_predictor.load_first_frame(frame_rgb)
    sam2_predictor.frame_idx = 0   # memory attention 거리 정상 유지
    bbox_arr = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=oid, bbox=bbox_arr)


# ============================================================
# Live 디스플레이
# ============================================================
def draw_tracking_overlay(frame_bgr, polygon, mask, yolo_bboxes,
                           crop_img, info_text, paused=False,
                           anchor_bbox=None):
    canvas = frame_bgr.copy()

    if polygon is not None and len(polygon) >= 3:
        cv2.polylines(canvas, [polygon], True, (0, 255, 255), 2)

    # YOLO bboxes — 사람 인식 컨텍스트 (참고용)
    for bb in yolo_bboxes:
        cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (0, 200, 0), 1)

    # SAM2 mask + bbox (magenta) — 추적 중인 사용자 객체
    if mask is not None and mask.any():
        color_layer = np.zeros_like(canvas)
        color_layer[mask] = (255, 0, 255)
        canvas = cv2.addWeighted(canvas, 1.0, color_layer, 0.45, 0)
        bb = get_bbox_from_mask(mask)
        if bb is not None:
            cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 255), 2)

    # 현재 클립의 anchor 박스 (시안색, locked crop window)
    if anchor_bbox is not None:
        x1, y1, x2, y2 = anchor_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 우상단 crop preview inset
    if crop_img is not None:
        H, W = canvas.shape[:2]
        ph, pw = crop_img.shape[:2]
        ox = W - pw - 10
        oy = 35
        if ox > 0 and oy + ph < H:
            canvas[oy:oy + ph, ox:ox + pw] = crop_img
            cv2.rectangle(canvas, (ox - 1, oy - 1),
                          (ox + pw + 1, oy + ph + 1), (255, 255, 255), 1)
            cv2.putText(canvas, "CROP", (ox, oy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, info_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if paused:
        h, w = canvas.shape[:2]
        cv2.putText(canvas, "PAUSED (SPACE/P:resume)", (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    return canvas


# ============================================================
# 한 영상 처리
# ============================================================
def _delete_file_if_exists(file_path):
    if file_path and os.path.exists(file_path):
        os.remove(file_path)


def _bbox_center_size(bbox):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * CROP_SIZE_PADDING)
    return cx, cy, size


def process_one_video(video_path, sam2_predictor, yolo_model):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    polygon = load_roi(video_path)
    if polygon is None:
        print(f"  [skip] ROI 없음 ({get_roi_path(video_path)})")
        return 'skip'

    action, start_idx, start_frame, init_mask, cap = select_object(
        video_path, polygon, sam2_predictor
    )
    if action == 'skip':
        return 'skip'
    if action == 'quit':
        return 'quit'

    print(f"  [select] start_frame={start_idx}")

    # 첫 anchor — 사용자가 시작 프레임에서 클릭한 객체 mask 의 bbox
    init_bbox = get_bbox_from_mask(init_mask)
    if init_bbox is None or init_mask.sum() < MIN_MASK_AREA:
        print("  [skip] 시작 mask 가 너무 작음")
        cap.release()
        return 'skip'

    cx, cy, locked_size = _bbox_center_size(init_bbox)
    last_bbox = init_bbox

    clip_index = 0
    clip_frame_count = 0
    last_anchor_frame = 0    # frame_count of 직전 anchor (start_frame = 0)
    suppress_intervention = False

    # 첫 클립 writer
    writer, file_path = open_clip_writer(video_name, USER_OBJ_ID, clip_index)

    # start_frame 첫 crop 저장
    if is_bbox_in_roi(init_bbox, polygon):
        crop_img = crop_with_padding(start_frame, cx, cy, locked_size)
        if crop_img is not None:
            writer.write(crop_img)
            clip_frame_count = 1

    track_window = (f"TRACKING {os.path.basename(video_path)}  "
                    "[SPACE/P:pause  N:skip-this  Q/ESC:quit-all]")
    cv2.namedWindow(track_window, cv2.WINDOW_NORMAL)

    def close_writer(min_frames):
        """writer release. clip_frame_count < min_frames 면 파일 삭제."""
        nonlocal writer, file_path
        if writer is None:
            return
        writer.release()
        if clip_frame_count < min_frames:
            _delete_file_if_exists(file_path)
        writer = None
        file_path = None

    paused = False
    frame_count = 0    # cap.read 이후 진행한 프레임 수

    try:
        while True:
            # 일시정지
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('p'), ord('P'), 32):
                    paused = False
                elif key in (ord('q'), ord('Q'), 27):
                    close_writer(min_frames=CROP_MIN_FRAMES_END)
                    cv2.destroyWindow(track_window)
                    print(f"\n  [quit] frame_count={frame_count}")
                    return 'quit'
                elif key in (ord('n'), ord('N')):
                    close_writer(min_frames=CROP_MIN_FRAMES_END)
                    cv2.destroyWindow(track_window)
                    print(f"\n  [skip] frame_count={frame_count}")
                    return 'done'
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1

            # ── SAM2 추적 ──
            obj_ids, mask_logits = sam2_predictor.track(frame_rgb)
            user_mask = None
            if USER_OBJ_ID in obj_ids:
                masks_all = (mask_logits > 0.0).squeeze(1).cpu().numpy()
                idx = obj_ids.index(USER_OBJ_ID)
                m = masks_all[idx]
                if m.sum() >= MIN_MASK_AREA:
                    user_mask = m

            # ── YOLO (display 컨텍스트용) ──
            results = yolo_model.predict(frame, verbose=False)
            yolo_bboxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bb = [int(x1), int(y1), int(x2), int(y2)]
                if not is_bbox_in_roi(bb, polygon):
                    continue
                yolo_bboxes.append(bb)

            # ── crop ──
            #   locked_size : 클립 안에서 고정 (anchor 시점 값)
            #   cx, cy      : 매 프레임 SAM2 mask 중심으로 EMA 추적 (객체 따라감)
            crop_img_for_display = None
            if user_mask is not None:
                bbox_now = get_bbox_from_mask(user_mask)
                if is_bbox_in_roi(bbox_now, polygon):
                    last_bbox = bbox_now
                    cx_raw = (bbox_now[0] + bbox_now[2]) / 2
                    cy_raw = (bbox_now[1] + bbox_now[3]) / 2
                    cx = CROP_EMA_ALPHA * cx_raw + (1 - CROP_EMA_ALPHA) * cx
                    cy = CROP_EMA_ALPHA * cy_raw + (1 - CROP_EMA_ALPHA) * cy
                if writer is not None:
                    crop_img_for_display = crop_with_padding(
                        frame, cx, cy, locked_size
                    )
                    if crop_img_for_display is not None:
                        writer.write(crop_img_for_display)
                        clip_frame_count += 1

            # ── 4초 강제 anchor UI ──
            need_intervention = (not suppress_intervention
                                 and frame_count - last_anchor_frame >= CROP_CYCLE_FRAMES)

            # ── Live 디스플레이 ──
            anchor_bbox_for_display = [
                int(cx - locked_size // 2), int(cy - locked_size // 2),
                int(cx + locked_size // 2), int(cy + locked_size // 2),
            ]
            elapsed_in_clip = frame_count - last_anchor_frame
            status = (f"frame {frame_count} (start={start_idx})  "
                      f"clip={clip_index} "
                      f"({clip_frame_count} frames, {elapsed_in_clip}/{CROP_CYCLE_FRAMES} elapsed)  "
                      f"{'[ANCHOR UI OFF]' if suppress_intervention else ''}")
            canvas = draw_tracking_overlay(
                frame, polygon, user_mask, yolo_bboxes,
                crop_img_for_display, status,
                paused=False, anchor_bbox=anchor_bbox_for_display,
            )
            cv2.imshow(track_window, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                close_writer(min_frames=CROP_MIN_FRAMES_END)
                cv2.destroyWindow(track_window)
                print(f"\n  [quit] frame_count={frame_count}")
                return 'quit'

            if key in (ord('n'), ord('N')):
                close_writer(min_frames=CROP_MIN_FRAMES_END)
                cv2.destroyWindow(track_window)
                print(f"\n  [skip] frame_count={frame_count}")
                return 'done'

            if key in (ord('p'), ord('P'), 32):
                paused = True

            # ── 4초 도달 시 anchor UI ──
            if need_intervention:
                # 현재 클립 닫기 (4초 채운 클립 — 무조건 keep)
                close_writer(min_frames=0)

                status_text = f"clip {clip_index} done. Anchor next 4s clip"
                ui_action, payload = anchor_ui(
                    frame, polygon, sam2_predictor,
                    USER_OBJ_ID, last_bbox, status_text,
                )

                if ui_action == 'quit':
                    cv2.destroyWindow(track_window)
                    print(f"\n  [quit] frame_count={frame_count}")
                    return 'quit'

                if ui_action == 'drop':
                    cv2.destroyWindow(track_window)
                    print(f"\n  [drop] frame_count={frame_count}")
                    return 'done'

                # recover / continue / extend — 모두 SAM2 resync 후 새 클립 시작
                if ui_action == 'recover':
                    new_bbox = payload
                    full_resync_single(sam2_predictor, frame_rgb, USER_OBJ_ID, new_bbox)
                    cx, cy, locked_size = _bbox_center_size(new_bbox)
                    last_bbox = new_bbox
                    print(f"\n  [recover] clip {clip_index + 1} anchor")
                elif ui_action == 'continue':
                    cur_bbox = (get_bbox_from_mask(user_mask)
                                if user_mask is not None else last_bbox)
                    if cur_bbox is None:
                        cv2.destroyWindow(track_window)
                        print(f"\n  [end] mask 없음")
                        return 'done'
                    full_resync_single(sam2_predictor, frame_rgb, USER_OBJ_ID, cur_bbox)
                    cx, cy, locked_size = _bbox_center_size(cur_bbox)
                    last_bbox = cur_bbox
                    print(f"\n  [continue] clip {clip_index + 1} anchor (kept)")
                elif ui_action == 'extend':
                    cur_bbox = (get_bbox_from_mask(user_mask)
                                if user_mask is not None else last_bbox)
                    if cur_bbox is not None:
                        full_resync_single(sam2_predictor, frame_rgb, USER_OBJ_ID, cur_bbox)
                        cx, cy, locked_size = _bbox_center_size(cur_bbox)
                        last_bbox = cur_bbox
                    suppress_intervention = True
                    print(f"\n  [extend] anchor UI 비활성화")

                clip_index += 1
                writer, file_path = open_clip_writer(video_name, USER_OBJ_ID, clip_index)
                clip_frame_count = 0
                last_anchor_frame = frame_count

            if frame_count %30 == 0:
                print(f"  frame {frame_count}", end='\r')

        # 영상 끝 — 마지막 partial 클립은 4초 못 채우면 삭제
        close_writer(min_frames=CROP_MIN_FRAMES_END)
        cv2.destroyWindow(track_window)
        print(f"\n  [done] frame_count={frame_count} (start={start_idx})")
        return 'done'
    finally:
        cap.release()


# ============================================================
# Main
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

    print("YOLO 로드 중...")
    yolo_model = YOLO(WEIGHT_PATH)

    print("SAM2 로드 중...")
    GlobalHydra.instance().clear()
    sam2_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'sam2_streaming', 'sam2', 'configs')
    with initialize_config_dir(config_dir=sam2_config_dir, version_base=None):
        sam2_predictor = build_sam2_camera_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT)

    videos = find_videos(VIDEOS_DIR)
    if not videos:
        print(f"영상 없음: {VIDEOS_DIR}")
        return

    os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)
    print(f"\n총 {len(videos)}개 영상 처리 시작 — 출력: {CROP_OUTPUT_DIR}/\n")

    for i, video_path in enumerate(videos, 1):
        name = os.path.basename(video_path)
        print(f"\n[{i}/{len(videos)}] {name}")
        try:
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                result = process_one_video(video_path, sam2_predictor, yolo_model)
            if result == 'quit':
                print("\n사용자 종료 요청 — 남은 영상 건너뜀")
                break
        except Exception as e:
            print(f"  [error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 영상 간 GPU 메모리 단편화 방지 + 누적 텐서 정리
            torch.cuda.empty_cache()

    cv2.destroyAllWindows()
    print("\n=== 모든 영상 처리 완료 ===")


if __name__ == "__main__":
    main()
