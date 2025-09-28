import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 경로 설정
# =========================
audio_folder = r"D:\MDO\heartbeat\Dataset"  # .wav 들이 있는 폴더
visual_folder = r"D:\MDO\heartbeat\real_data_visualization\visual+99"  # 결과 저장 폴더
label_csv_path = r"D:\MDO\heartbeat\Dataset\REFERENCE.csv"  # CSV 경로

os.makedirs(visual_folder, exist_ok=True)

# =========================
# CSV 로드 (헤더 없음, 두 컬럼만)
# =========================
df = pd.read_csv(label_csv_path, header=None, names=["filename", "label"])
# 확장자 없는 이름만 키로 사용
df["key"] = df["filename"].apply(lambda x: os.path.splitext(str(x))[0])
label_map = dict(zip(df["key"], df["label"]))

print(f"[INFO] {len(label_map)}개의 라벨 로드 완료")

# =========================
# 파라미터
# =========================
stride = 99  # 0, 99, 198, ...처럼 '99씩 점프' (100칸 간격과 동일한 효과)
# 0, 100, 200, ... 원하시면 stride=100 으로 변경


# =========================
# 오디오 파일 시각화
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

for audio_path in audio_files:
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    label_val = label_map.get(file_name, "N/A")

    print(f"Processing {file_name} (label={label_val})")

    # 오디오 로드
    audio_sample, sampling_rate = librosa.load(audio_path, sr=None)

    sampled = audio_sample[::stride]

    if len(sampled) < 3:
        print(f"⚠ 건너뜀: {file_name} (너무 짧음)")
        continue

    offset = 1
    x = sampled[:-offset]
    y = sampled[offset:]

    # 저장 경로
    save_path = os.path.join(visual_folder, f"{file_name}.png")

    # 그리기
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=4, alpha=0.6)
    plt.title(f"{file_name} | Label: {label_val} | SR={sampling_rate}")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+1]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("✅ 모든 wav 파일 시각화 완료! 라벨이 포함되어 저장되었습니다.")
