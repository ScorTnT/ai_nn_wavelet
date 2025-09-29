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
visual_folder = r"data_visualization/n_n2_cnt399_8sec"  # 결과 저장 폴더
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
offset = 2   # n, n+2 방식
target_duration = 8.0  # 2초 목표 길이
SKIP_SAMPLES = 20  # 매 20개마다 하나씩만 사용

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
    
    # 실제 길이로 자르기
    target_samples = int(actual_duration * sampling_rate)
    audio_sample = audio_sample[:target_samples]

    # stride 제거 - 모든 샘플 사용
    if len(audio_sample) < offset + 1:
        print(f"⚠ 건너뜀: {file_name} (너무 짧음)")
        continue

    audio_sample = audio_sample[::SKIP_SAMPLES]
    print(f"  서브샘플링 후 길이: {len(audio_sample)} 샘플")

    x = audio_sample[:-offset]
    y = audio_sample[offset:]
    
    # 홀수 인덱스와 짝수 인덱스 분리
    odd_indices = np.arange(0, len(x), 2)    # 0, 2, 4, 6, ... (실제로는 홀수번째 샘플)
    even_indices = np.arange(1, len(x), 2)   # 1, 3, 5, 7, ... (실제로는 짝수번째 샘플)
    
    x_odd = x[odd_indices]
    y_odd = y[odd_indices]
    x_even = x[even_indices]
    y_even = y[even_indices]

    print(f"  홀수 점 개수: {len(x_odd)}, 짝수 점 개수: {len(x_even)}")

    # 홀수 인덱스만 따로 저장 (빨간색)
    save_path_odd = os.path.join(visual_folder, f"{file_name}_n_n+2_odd.png")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_odd, y_odd, s=6, alpha=0.8, c='red', label=f'홀수 인덱스 ({len(x_odd)} points)')
    
    plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+2 | 홀수 인덱스만")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+2]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path_odd, dpi=300)
    plt.close()
    
    print(f"  → {save_path_odd} 저장 완료")

    # 짝수 인덱스만 따로 저장 (파란색)
    save_path_even = os.path.join(visual_folder, f"{file_name}_n_n+2_even.png")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_even, y_even, s=6, alpha=0.8, c='blue', label=f'짝수 인덱스 ({len(x_even)} points)')
    
    plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+2 | 짝수 인덱스만")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+2]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path_even, dpi=300)
    plt.close()
    
    print(f"  → {save_path_even} 저장 완료")

    # 추가로 통합 이미지도 저장 (홀수+짝수)
    save_path_combined = os.path.join(visual_folder, f"{file_name}_n_n+2_combined.png")
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(x_odd, y_odd, s=4, alpha=0.7, c='blue', label=f'홀수 ({len(x_odd)} points)')
    plt.scatter(x_even, y_even, s=4, alpha=0.7, c='blue', label=f'짝수 ({len(x_even)} points)')
    
    plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+2 | 전체")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+2]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path_combined, dpi=300)
    plt.close()
    
    print(f"  → {save_path_combined} 저장 완료")

print("✅ 홀수/짝수 분리 시각화 완료!")
print("   파일 저장 규칙:")
print("   - _odd.png: 홀수 인덱스만 (빨간색)")
print("   - _even.png: 짝수 인덱스만 (파란색)")  
print("   - _combined.png: 홀수+짝수 함께")