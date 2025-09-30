import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 경로 설정
# =========================
audio_folder = r"training"  # .wav 들이 있는 폴더
visual_folder = r"data_visualization/n_n2"  # 결과 저장 폴더
label_csv_path = r"training/REFERENCE.csv"  # CSV 경로

os.makedirs(visual_folder, exist_ok=True)

# =========================
# CSV 로드 (REFERENCE.csv 형식: filename,label)
# =========================
df = pd.read_csv(label_csv_path, header=None, names=["filename", "label"])
# 확장자 없는 이름만 키로 사용
df["key"] = df["filename"].apply(lambda x: os.path.splitext(str(x))[0])
label_map = dict(zip(df["key"], df["label"]))

# 라벨별로 파일 분류 (1과 -1 각각 dataSetCnt개씩만 선택)
dataSetCnt = 7
label_1_files = [k for k, v in label_map.items() if v == 1][:dataSetCnt]
label_minus1_files = [k for k, v in label_map.items() if v == -1][:dataSetCnt]
selected_files = label_1_files + label_minus1_files

print(f"[INFO] 라벨 1인 파일 {dataSetCnt}개: {label_1_files}")
print(f"[INFO] 라벨 -1인 파일 {dataSetCnt}개: {label_minus1_files}")
print(f"[INFO] 총 {len(selected_files)}개 파일 선택됨")

# =========================
# 파라미터
# =========================
offset = 2   # n, n+{offset} 방식
target_duration = 8.0  # 8초 목표 길이

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
    audio_sample, sampling_rate = librosa.load(audio_path, sr=None)
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
    audio_sample, sampling_rate = librosa.load(audio_path, sr=None)
    print(f" - 샘플링 레이트: {sampling_rate}, 총 샘플 수: {len(audio_sample)}, 길이: {len(audio_sample)/sampling_rate:.2f}초")
    # 실제 길이로 자르기
    target_samples = int(actual_duration * sampling_rate)
    audio_sample = audio_sample[:target_samples]

    # stride 제거 - 모든 샘플 사용
    if len(audio_sample) < offset + 1:
        print(f"⚠ 건너뜀: {file_name} (너무 짧음)")
        continue

    x = audio_sample[:-offset]
    y = audio_sample[offset:]
    
    # 홀수 인덱스와 짝수 인덱스 분리
    odd_indices = np.arange(0, len(x), 2)    # 0, 2, 4, 6, ... (실제로는 홀수번째 샘플)
    even_indices = np.arange(1, len(x), 2)   # 1, 3, 5, 7, ... (실제로는 짝수번째 샘플)
    
    x_odd = x[odd_indices]
    y_odd = y[odd_indices]
    x_even = x[even_indices]
    y_even = y[even_indices]

    # 저장 경로
    save_path = os.path.join(visual_folder, f"{file_name}_n_n+2.png")

    # 그리기
    plt.figure(figsize=(10, 8))
    
    # 홀수 인덱스 점들 (빨간색)
    plt.scatter(x_odd, y_odd, s=4, alpha=0.7, c='red', label='(1,3,5,...)')
    
    # 짝수 인덱스 점들 (파란색)  
    plt.scatter(x_even, y_even, s=4, alpha=0.7, c='blue', label='(2,4,6,...)')
    
    plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+2")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+2]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print("✅ 선택된 wav 파일들 시각화 완료! 홀수/짝수 인덱스가 다른 색으로 표시되어 저장되었습니다.")