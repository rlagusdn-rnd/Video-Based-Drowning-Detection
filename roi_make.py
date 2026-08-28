"""roi_make.py
폴더 내 모든 영상의 첫 프레임에서 ROI 폴리곤을 마우스로 입력받아 json으로 저장.

저장 규칙:
  영상 옆 같은 이름 + .roi.json
  예: datasets/videos/swim_70.mp4 → datasets/videos/swim_70.roi.json

키 매핑:
  q   : 현재 영상 ROI 저장 + 다음
  n   : 저장 안 하고 다음
  r   : 점들 리셋 (다시 그리기)
  Esc : 전체 종료
"""

import cv2
import os
import json
import glob

VIDEOS_DIR       = 'datasets/videos/'
VIDEO_EXTENSIONS = ['.mp4']

# display 최대 크기 — 영상이 이보다 크면 화면에 맞게 축소해서 보여줌
# (저장되는 좌표는 항상 원본 frame 좌표계)
MAX_DISPLAY_HEIGHT = 900
MAX_DISPLAY_WIDTH  = 1600


def find_videos(videos_dir):
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(glob.glob(os.path.join(videos_dir, f'*{ext}')))
    return sorted(files)


def get_roi_path(video_path):
    """영상 옆 같은 이름의 .roi.json 경로"""
    return os.path.splitext(video_path)[0] + '.roi.json'


def make_roi_for_video(video_path):
    """한 영상에 대해 ROI 마우스 입력 → 저장. 반환: 'saved' | 'skipped' | 'aborted'"""
    cap = cv2.VideoCapture(video_path)
    ret, original = cap.read()
    cap.release()
    if not ret:
        print(f"  [error] 첫 프레임 읽기 실패")
        return 'skipped'

    # display scale 계산 — 영상이 모니터보다 크면 축소 (1.0 cap: 작은 영상은 확대 안 함)
    H, W = original.shape[:2]
    scale = min(MAX_DISPLAY_HEIGHT / H, MAX_DISPLAY_WIDTH / W, 1.0)
    if scale < 1.0:
        disp_W = int(W * scale)
        disp_H = int(H * scale)
        original_disp = cv2.resize(original, (disp_W, disp_H))
        print(f"  [info] {W}x{H} → {disp_W}x{disp_H} display (scale={scale:.3f}); ROI 좌표는 원본 기준 저장")
    else:
        original_disp = original
        print(f"  [info] {W}x{H} (display 크기 그대로)")

    frame = original_disp.copy()
    points = []   # 원본 frame 좌표계

    def redraw():
        nonlocal frame
        frame = original_disp.copy()
        # 원본 좌표 → display 좌표 변환 후 그리기
        disp_points = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
        for p in disp_points:
            cv2.circle(frame, p, 5, (0, 255, 0), -1)
        if len(disp_points) > 1:
            for i in range(1, len(disp_points)):
                cv2.line(frame, disp_points[i-1], disp_points[i], (0, 255, 0), 2)
            if len(disp_points) > 2:
                cv2.line(frame, disp_points[-1], disp_points[0], (0, 255, 0), 2)

    def callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # display 좌표 → 원본 좌표 변환 후 저장
            ox = int(round(x / scale))
            oy = int(round(y / scale))
            points.append((ox, oy))
            redraw()

    title = f"ROI: {os.path.basename(video_path)}  [q=save / n=skip / r=reset / Esc=quit]"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, callback)

    while True:
        cv2.imshow(title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if len(points) >= 3:
                roi_path = get_roi_path(video_path)
                with open(roi_path, 'w') as f:
                    json.dump({"points": points}, f, indent=2)
                print(f"  [saved] {roi_path}  ({len(points)}개 점)")
                cv2.destroyWindow(title)
                return 'saved'
            print("  [warn] 점 3개 이상 필요. 더 찍거나 'n'으로 skip")
        elif key == ord('n'):
            print("  [skipped]")
            cv2.destroyWindow(title)
            return 'skipped'
        elif key == ord('r'):
            points.clear()
            redraw()
        elif key == 27:  # Esc
            cv2.destroyWindow(title)
            return 'aborted'


def main():
    videos = find_videos(VIDEOS_DIR)
    if not videos:
        print(f"영상 없음: {VIDEOS_DIR}")
        return

    print(f"총 {len(videos)}개 영상 발견 ({VIDEOS_DIR})\n")

    saved, skipped, already = 0, 0, 0
    for i, video_path in enumerate(videos, 1):
        name = os.path.basename(video_path)
        roi_path = get_roi_path(video_path)
        if os.path.exists(roi_path):
            already += 1
            print(f"[{i}/{len(videos)}] {name} — ROI 이미 있음, skip")
            continue

        print(f"[{i}/{len(videos)}] {name}")
        result = make_roi_for_video(video_path)
        if result == 'saved':
            saved += 1
        elif result == 'skipped':
            skipped += 1
        elif result == 'aborted':
            print("\n[abort] 사용자 종료")
            break

    print(f"\n=== 완료 ===")
    print(f"  저장됨   : {saved}")
    print(f"  skip됨   : {skipped}")
    print(f"  기존 존재: {already}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
