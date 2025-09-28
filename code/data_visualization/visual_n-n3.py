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
visual_folder = r"data_visualization/n_n3_separated"  # 결과 저장 폴더
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
dataSetCnt = 4
label_1_files = [k for k, v in label_map.items() if v == 1][:dataSetCnt]
label_minus1_files = [k for k, v in label_map.items() if v == -1][:dataSetCnt]
selected_files = label_1_files + label_minus1_files

print(f"[INFO] 라벨 1인 파일 {dataSetCnt}개: {label_1_files}")
print(f"[INFO] 라벨 -1인 파일 {dataSetCnt}개: {label_minus1_files}")
print(f"[INFO] 총 {len(selected_files)}개 파일 선택됨")

# =========================
# 파라미터
# =========================
offset = 3   # n, n+3 방식
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
    
    # 실제 길이로 자르기
    target_samples = int(actual_duration * sampling_rate)
    audio_sample = audio_sample[:target_samples]

    # n, n+3 방식
    if len(audio_sample) < offset + 1:
        print(f"⚠ 건너뜀: {file_name} (너무 짧음)")
        continue

    x = audio_sample[:-offset]
    y = audio_sample[offset:]
    
    # 3개 그룹으로 인덱스 분리 (offset=3이므로 3개 그룹)
    group_0_indices = np.arange(0, len(x), 3)    # 0, 3, 6, 9, ... (첫 번째 그룹)
    group_1_indices = np.arange(1, len(x), 3)    # 1, 4, 7, 10, ... (두 번째 그룹)
    group_2_indices = np.arange(2, len(x), 3)    # 2, 5, 8, 11, ... (세 번째 그룹)
    
    x_group_0 = x[group_0_indices]
    y_group_0 = y[group_0_indices]
    x_group_1 = x[group_1_indices]
    y_group_1 = y[group_1_indices]
    x_group_2 = x[group_2_indices]
    y_group_2 = y[group_2_indices]

    # 각 그룹별로 개별 이미지 생성
    groups_data = [
        (x_group_0, y_group_0, 'red', '_r', '그룹 0 (0,3,6,...)'),
        (x_group_1, y_group_1, 'blue', '_b', '그룹 1 (1,4,7,...)'),
        (x_group_2, y_group_2, 'green', '_g', '그룹 2 (2,5,8,...)')
    ]
    
    # 각 색상별로 별도 파일 저장
    for x_data, y_data, color, suffix, label_text in groups_data:
        save_path = os.path.join(visual_folder, f"{file_name}_n_n+3{suffix}.png")
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(x_data, y_data, s=4, alpha=0.7, c=color, label=label_text)
        
        plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+3 | {label_text}")
        plt.xlabel("Sample[n]")
        plt.ylabel("Sample[n+3]")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"  → {save_path} 저장 완료")

    # 추가로 모든 그룹이 함께 있는 통합 이미지도 저장
    save_path_combined = os.path.join(visual_folder, f"{file_name}_n_n+3_combined.png")
    
    plt.figure(figsize=(10, 8))
    
    plt.scatter(x_group_0, y_group_0, s=4, alpha=0.7, c='red', label=' 0 (0,3,6,...)')
    plt.scatter(x_group_1, y_group_1, s=4, alpha=0.7, c='blue', label=' 1 (1,4,7,...)')
    plt.scatter(x_group_2, y_group_2, s=4, alpha=0.7, c='green', label=' 2 (2,5,8,...)')
    
    plt.title(f"{file_name} | Label: {label_val} | Duration: {actual_duration:.2f}s | n,n+3 | All Groups")
    plt.xlabel("Sample[n]")
    plt.ylabel("Sample[n+3]")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path_combined, dpi=300)
    plt.close()
    
    print(f"  → {save_path_combined} 저장 완료")

print("✅ 선택된 wav 파일들 n,n+3 시각화 완료!")
print("   각 색상별로 개별 파일과 통합 파일이 저장되었습니다.")
print("   파일 이름 규칙:")
print("   - _r: 빨간색 그룹 (0,3,6,...)")
print("   - _b: 파란색 그룹 (1,4,7,...)")
print("   - _g: 녹색 그룹 (2,5,8,...)")
print("   - _combined: 모든 그룹 함께")