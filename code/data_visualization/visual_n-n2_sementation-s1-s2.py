import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 경로 설정
# =========================
audio_folder = r"validation"  # .wav 들이 있는 폴더
visual_folder = r"data_visualization/n_n1_seg_8sec"  # 결과 저장 폴더
label_csv_path = r"validation/REFERENCE.csv"  # CSV 경로

os.makedirs(visual_folder, exist_ok=True)

# =========================
# CSV 로드 (REFERENCE.csv 형식: filename,label)
# =========================
df = pd.read_csv(label_csv_path, header=None, names=["filename", "label"])
# 확장자 없는 이름만 키로 사용
df["key"] = df["filename"].apply(lambda x: os.path.splitext(str(x))[0])
label_map = dict(zip(df["key"], df["label"]))

# 라벨별로 파일 분류 (1과 -1 각각 dataSetCnt개씩만 선택)
dataSetCnt = 3
label_1_files = [k for k, v in label_map.items() if v == 1][:dataSetCnt]
label_minus1_files = [k for k, v in label_map.items() if v == -1][:dataSetCnt]
selected_files = label_1_files + label_minus1_files

print(f"[INFO] 라벨 1인 파일 {dataSetCnt}개: {label_1_files}")
print(f"[INFO] 라벨 -1인 파일 {dataSetCnt}개: {label_minus1_files}")
print(f"[INFO] 총 {len(selected_files)}개 파일 선택됨")

# =========================
# 파라미터
# =========================
offset = 1   # n, n+1 방식
target_duration = 8.0  # 8초 목표 길이
segment_size = 1024  # 각 세그먼트 크기
target_sampling_rate = 2000  # 목표 샘플링 레이트


# =========================
# 오디오 파일 시각화
# =========================
audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))

# 선택된 파일들만 필터링
selected_audio_files = []
for audio_path in audio_files:
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    if file_name in selected_files:
        selected_audio_files.append(audio_path)

# 모든 파일의 길이 확인하여 최소 길이 찾기
min_duration = float('inf')
audio_durations = {}

for audio_path in selected_audio_files:
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    audio_sample, sampling_rate = librosa.load(audio_path, sr=target_sampling_rate)
    duration = len(audio_sample) / sampling_rate
    audio_durations[file_name] = duration
    min_duration = min(min_duration, duration)

# 실제 사용할 길이 결정 (8초 또는 최소 길이 중 작은 값)
actual_duration = min(target_duration, min_duration)
print(f"[INFO] 실제 사용할 오디오 길이: {actual_duration:.2f}초")

for audio_path in selected_audio_files:
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    label_val = label_map.get(file_name, "N/A")

    print(f"Processing {file_name} (label={label_val})")

    # 오디오 로드
    audio_sample, sampling_rate = librosa.load(audio_path, sr=target_sampling_rate)
    
    # 실제 길이로 자르기 (8초 * target_sampling_rate = target_samples)
    target_samples = int(actual_duration * sampling_rate)
    audio_sample = audio_sample[:target_samples]

    # 1024개씩 세그먼트로 나누기
    num_segments = target_samples // segment_size  # target_samples // 1024
    print(f"  총 세그먼트 개수: {num_segments} (샘플링 레이트: {sampling_rate}Hz)")

    for segment_idx in range(num_segments):
        start_idx = segment_idx * segment_size
        end_idx = start_idx + segment_size
        segment_data = audio_sample[start_idx:end_idx]
        
        # n, n+1 방식으로 x, y 좌표 생성
        x = segment_data[:-offset]  # n
        y = segment_data[offset:]   # n+1
        
        print(f"  세그먼트 {segment_idx+1}: {len(x)} 점들")

        # 세그먼트별 이미지 저장
        save_path = os.path.join(visual_folder, f"{file_name}_segment_{segment_idx+1:02d}.png")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, s=6, alpha=0.8, c='blue', label=f'Segment {segment_idx+1} ({len(x)} points)')
        
        plt.title(f"{file_name} | Label: {label_val} | Segment: {segment_idx+1}/{num_segments} | n,n+1")
        plt.xlabel("Sample[n]")
        plt.ylabel("Sample[n+1]")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
        plt.legend()
        plt.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"    → {save_path} 저장 완료")

print("✅ 세그먼트별 시각화 완료!")
print("   파일 저장 규칙:")
print("   - _segment_01.png ~ _segment_15.png: 각 세그먼트별 n,n+1 scatter plot (파란색)")