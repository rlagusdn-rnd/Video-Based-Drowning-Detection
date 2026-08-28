# 비디오 클립 → 16프레임 numpy 배열로 만들어주는 Dataset 클래스
# Stage 1 (feature 추출) 단계에서 사용

from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    # 폴더명 → 라벨 ID (many-to-one 매핑 허용: 여러 폴더가 같은 라벨로 묶일 수 있음)
    FOLDER2ID = {
        "drowning": 0,
        "floating": 1,
        "standing": 1,
        "swimming": 1,
        "complex":  1,
    }
    # 라벨 ID → 출력/표시용 이름
    ID2NAME = {0: "drowning", 1: "normal"}
    # 데이터 수가 적거나 의도적으로 학습/평가에서 빼는 폴더
    EXCLUDE_FOLDERS = {"submerged"}

    def __init__(self, root, num_frames=16):
        """
        Args:
            root: split 폴더 경로 (예: "datasets/train")
            num_frames: 비디오 1개에서 뽑을 프레임 수
        """
        # 한번만 초기화 되는 부분
        self.root = Path(root)
        self.num_frames = num_frames
        self.samples = []  # (video_path, label_id, folder_name) 튜플 저장

        # root 안의 클래스 폴더를 직접 순회 (FOLDER2ID에 있는 것만, EXCLUDE는 스킵)
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            folder = class_dir.name
            if folder in self.EXCLUDE_FOLDERS:
                print(f"[VideoDataset] EXCLUDE: {folder} (스킵)")
                continue
            if folder not in self.FOLDER2ID:
                print(f"[VideoDataset] Warning: 매핑 없는 폴더 무시: {folder}")
                continue
            label_id = self.FOLDER2ID[folder]
            for video_path in sorted(class_dir.glob("*.mp4")):
                self.samples.append((video_path, label_id, folder))

        # 디버깅용: 라벨별 + 폴더별 카운트 출력
        print(f"[VideoDataset] root={root}, total={len(self.samples)}")
        for label_id, name in self.ID2NAME.items():
            count = sum(1 for _, lbl, _ in self.samples if lbl == label_id)
            print(f"  [{label_id}] {name}: {count}")
        # 어떤 폴더가 어느 라벨에 흡수됐는지 보여줌 (디버깅용)
        from collections import Counter
        folder_counts = Counter(f for _, _, f in self.samples)
        for folder, cnt in folder_counts.items():
            print(f"    └ {folder} → {self.ID2NAME[self.FOLDER2ID[folder]]}: {cnt}")

    def __len__(self):
        # 데이터셋의 크기를 반환
        return len(self.samples)

    def __getitem__(self, idx):
        # 매 샘플마다 호출되는 부분
        path, label, folder = self.samples[idx]

        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)  # 비디오 전체 프레임 수

        # num_frames 개수만큼 프레임 균일 샘플링 된 수 저장
        indices = np.linspace(0, total - 1, self.num_frames).astype(int)
        # 샘플링 된 프레임
        frames = vr.get_batch(indices.tolist()).asnumpy()

        video_name = path.stem  # 예: "swim_12_id_0001_clip_000"

        # folder는 features 저장 시 원본 폴더명 보존을 위해 함께 반환
        return frames, label, video_name, folder
