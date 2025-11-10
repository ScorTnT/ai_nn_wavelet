import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 파라미터
# =========================
offset = 1   # n, n+1 방식
target_duration = 8.0  # 8초 목표 길이
skip_duration = 0    # 초반 건너뛸 시간 (초)
segment_size = 0  # 각 세그먼트 크기 (0이면 2차 segmentation 비활성화)
target_sampling_rate = 200  # 목표 샘플링 레이트
dataSetCnt = 3 # 변환할 파일 개수 (라벨 1과 -1 각각)

# 심박 segmentation 파라미터
heart_rate_bpm = 80  # 예상 심박수 (분당 박동수)
systole_ratio = 0.35  # 수축기 비율 (전체 심박 주기 중) S1 ratio
diastole_ratio = 0.65  # 이완기 비율 (전체 심박 주기 중) S2 ratio

# =========================
# 경로 설정
# =========================
audio_folder = r"validation"  # .wav 들이 있는 폴더
visual_folder = r"_data_visualization/n_n1-seg_s1_s2/"  # 결과 저장 폴더
visual_folder += (f"{heart_rate_bpm}-{systole_ratio}_{skip_duration}-{segment_size}-{target_sampling_rate}")
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
label_1_files = [k for k, v in label_map.items() if v == 1][:dataSetCnt]
label_minus1_files = [k for k, v in label_map.items() if v == -1][:dataSetCnt]
selected_files = label_1_files + label_minus1_files

print(f"[INFO] 라벨 1인 파일 {dataSetCnt}개: {label_1_files}")
print(f"[INFO] 라벨 -1인 파일 {dataSetCnt}개: {label_minus1_files}")
print(f"[INFO] 총 {len(selected_files)}개 파일 선택됨")

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
    
    # skip_duration 적용 (초반 건너뛰기)
    skip_samples = int(skip_duration * sampling_rate)  # 건너뛸 샘플 수
    target_samples = int(actual_duration * sampling_rate)  # 사용할 샘플 수
    
    # 전체 길이 확인
    total_needed_samples = skip_samples + target_samples
    if len(audio_sample) < total_needed_samples:
        print(f"⚠ 건너뜀: {file_name} (길이 부족: {len(audio_sample)} < {total_needed_samples})")
        continue
    
    # skip_duration초 건너뛰고 그 이후 target_duration초 사용
    audio_sample = audio_sample[skip_samples:skip_samples + target_samples]
    
    # =========================
    # 1차 Segmentation: 심박 주기 기반 (수축기/이완기)
    # =========================
    heart_cycle_duration = 60.0 / heart_rate_bpm  # 초 단위 심박 주기
    heart_cycle_samples = int(heart_cycle_duration * sampling_rate)  # 샘플 단위
    
    systole_samples = int(heart_cycle_samples * systole_ratio)  # 수축기 샘플 수
    diastole_samples = int(heart_cycle_samples * diastole_ratio)  # 이완기 샘플 수
    
    num_heart_cycles = len(audio_sample) // heart_cycle_samples
    print(f"  심박 주기: {heart_cycle_duration:.3f}초 ({heart_cycle_samples} 샘플)")
    print(f"  수축기: {systole_samples} 샘플, 이완기: {diastole_samples} 샘플")
    print(f"  총 심박 주기 개수: {num_heart_cycles}")

    for cycle_idx in range(num_heart_cycles):
        cycle_start = cycle_idx * heart_cycle_samples
        cycle_end = cycle_start + heart_cycle_samples
        
        if cycle_end > len(audio_sample):
            break
            
        cycle_data = audio_sample[cycle_start:cycle_end]
        
        # 수축기 (S1) 구간
        systole_data = cycle_data[:systole_samples]
        # 이완기 (S2) 구간  
        diastole_data = cycle_data[systole_samples:systole_samples + diastole_samples]
        
        # 수축기와 이완기 각각 처리
        for phase_name, phase_data in [("S1", systole_data), ("S2", diastole_data)]:
            if len(phase_data) < offset + 1:
                continue
                
            # =========================
            # 2차 Segmentation: 기존 방식 (segment_size > 0인 경우만)
            # =========================
            if segment_size > 0 and len(phase_data) >= segment_size:
                # 2차 segmentation 적용
                num_segments = len(phase_data) // segment_size
                print(f"    {phase_name} - 2차 세그먼트 개수: {num_segments}")
                
                for segment_idx in range(num_segments):
                    seg_start = segment_idx * segment_size
                    seg_end = seg_start + segment_size
                    segment_data = phase_data[seg_start:seg_end]
                    
                    # n, n+1 방식으로 x, y 좌표 생성
                    x = segment_data[:-offset]
                    y = segment_data[offset:]
                    
                    if len(x) == 0:
                        continue
                    
                    # 파일명: cycle_phase_segment
                    save_path = os.path.join(visual_folder, 
                        f"{file_name}_cycle{cycle_idx+1:02d}_{phase_name}_seg{segment_idx+1:02d}.png")
                    
                    plt.figure(figsize=(10, 8))
                    plt.scatter(x, y, s=6, alpha=0.8, c='blue', 
                               label=f'Cycle{cycle_idx+1} {phase_name} Seg{segment_idx+1} ({len(x)} points)')
                    
                    plt.title(f"{file_name} | Label: {label_val} | Cycle{cycle_idx+1} {phase_name} Seg{segment_idx+1} | n,n+{offset}")
                    plt.xlabel("Sample[n]")
                    plt.ylabel(f"Sample[n+{offset}]")
                    plt.xlim(-1, 1)
                    plt.ylim(-1, 1)
                    
                    plt.legend()
                    plt.grid(True, alpha=0.2)
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                    
                    print(f"      → {save_path} 저장 완료")
            
            else:
                # 2차 segmentation 없이 전체 phase 데이터 사용
                x = phase_data[:-offset]
                y = phase_data[offset:]
                
                if len(x) == 0:
                    continue
                
                save_path = os.path.join(visual_folder, 
                    f"{file_name}_cycle{cycle_idx+1:02d}_{phase_name}.png")
                
                plt.figure(figsize=(10, 8))
                plt.scatter(x, y, s=6, alpha=0.8, c='blue', 
                           label=f'Cycle{cycle_idx+1} {phase_name} ({len(x)} points)')
                
                plt.title(f"{file_name} | Label: {label_val} | Cycle{cycle_idx+1} {phase_name} | n,n+{offset}")
                plt.xlabel("Sample[n]")
                plt.ylabel(f"Sample[n+{offset}]")
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                
                plt.legend()
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                
                print(f"    → {save_path} 저장 완료")

print("✅ 심박 주기 기반 세그먼트별 시각화 완료!")
print("   파일 저장 규칙:")
print("   - _cycle##_S1_systole_seg##.png: 수축기 2차 세그먼트 (segment_size > 0인 경우)")
print("   - _cycle##_S2_diastole_seg##.png: 이완기 2차 세그먼트 (segment_size > 0인 경우)")
print("   - _cycle##_S1_systole.png: 수축기 전체 (segment_size = 0인 경우)")
print("   - _cycle##_S2_diastole.png: 이완기 전체 (segment_size = 0인 경우)")
print(f"   심박수: {heart_rate_bpm}bpm, 수축기 비율: {systole_ratio}, 이완기 비율: {diastole_ratio}")