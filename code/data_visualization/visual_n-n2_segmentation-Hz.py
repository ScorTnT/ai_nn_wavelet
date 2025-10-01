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
dataSetCnt = 5 # 변환할 파일 개수 (라벨 1과 -1 각각)

# 주파수 기반 segmentation 파라미터
s1_freq_low = 20    # S1 저주파 대역 (Hz)
s1_freq_high = 200  # S1 고주파 대역 (Hz)   # 보통 40~100
s2_freq_low = 50    # S2 저주파 대역 (Hz)  
s2_freq_high = 400  # S2 고주파 대역 (Hz)   # 보통 80~200 # + 고주파 잡음 처리
noise_freq_high = 20  # 저주파 잡음 차단 (Hz) # 저주파 잡음 처리

# =========================
# 경로 설정
# =========================
audio_folder = r"validation"  # .wav 들이 있는 폴더
visual_folder = r"_data_visualization/n_n1-seg_Hz"  # 결과 저장 폴더
visual_folder += (f"_{skip_duration}-{segment_size}-{target_sampling_rate}")
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
    # 1차 Segmentation: 주파수 대역 기반 필터링
    # =========================
    
    def apply_bandpass_filter(signal, low_freq, high_freq, sampling_rate):
        """주파수 대역 통과 필터 적용"""
        # FFT
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # 주파수 마스크 생성
        mask = np.zeros(len(freqs), dtype=bool)
        mask[(np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)] = True
        
        # 필터 적용
        fft_filtered = fft_signal * mask
        
        # IFFT로 시간 도메인으로 복원
        filtered_signal = np.real(np.fft.ifft(fft_filtered))
        return filtered_signal
    
    # 각 주파수 대역별로 신호 추출
    print(f"  원본 신호 길이: {len(audio_sample)} 샘플")
    
    # S1 대역 (20-200Hz) - 수축기음 특성
    s1_signal = apply_bandpass_filter(audio_sample, s1_freq_low, s1_freq_high, sampling_rate)
    print(f"  S1 대역 ({s1_freq_low}-{s1_freq_high}Hz) 추출 완료")
    
    # S2 대역 (50-400Hz) - 이완기음 특성  
    s2_signal = apply_bandpass_filter(audio_sample, s2_freq_low, s2_freq_high, sampling_rate)
    print(f"  S2 대역 ({s2_freq_low}-{s2_freq_high}Hz) 추출 완료")
    
    # 저주파 잡음 제거된 전체 신호 (20Hz 이상)
    clean_signal = apply_bandpass_filter(audio_sample, noise_freq_high, sampling_rate//2, sampling_rate)
    print(f"  Clean 신호 ({noise_freq_high}Hz 이상) 추출 완료")
    
    # 각 주파수 대역별 신호 처리
    freq_bands = [
        ("S1_band", s1_signal, f"{s1_freq_low}-{s1_freq_high}Hz"),
        ("S2_band", s2_signal, f"{s2_freq_low}-{s2_freq_high}Hz"), 
        ("Clean_band", clean_signal, f"{noise_freq_high}Hz+")
    ]
    
    for band_name, band_signal, freq_desc in freq_bands:
        if len(band_signal) < offset + 1:
            continue
            
        print(f"  처리 중: {band_name} ({freq_desc})")
        
        # =========================
        # 2차 Segmentation: 기존 방식 (segment_size > 0인 경우만)
        # =========================
        if segment_size > 0 and len(band_signal) >= segment_size:
            # 2차 segmentation 적용
            num_segments = len(band_signal) // segment_size
            print(f"    {band_name} - 2차 세그먼트 개수: {num_segments}")
            
            for segment_idx in range(num_segments):
                seg_start = segment_idx * segment_size
                seg_end = seg_start + segment_size
                segment_data = band_signal[seg_start:seg_end]
                
                # n, n+1 방식으로 x, y 좌표 생성
                x = segment_data[:-offset]
                y = segment_data[offset:]
                
                if len(x) == 0:
                    continue
                
                # 파일명: band_segment
                save_path = os.path.join(visual_folder, 
                    f"{file_name}_{band_name}_seg{segment_idx+1:02d}.png")
                
                plt.figure(figsize=(10, 8))
                plt.scatter(x, y, s=6, alpha=0.8, c='blue', 
                           label=f'{band_name} Seg{segment_idx+1} ({len(x)} points)')
                
                plt.title(f"{file_name} | Label: {label_val} | {band_name} ({freq_desc}) Seg{segment_idx+1} | n,n+{offset}")
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
            # 2차 segmentation 없이 전체 대역 데이터 사용
            x = band_signal[:-offset]
            y = band_signal[offset:]
            
            if len(x) == 0:
                continue
            
            save_path = os.path.join(visual_folder, 
                f"{file_name}_{band_name}.png")
            
            plt.figure(figsize=(10, 8))
            plt.scatter(x, y, s=6, alpha=0.8, c='blue', 
                       label=f'{band_name} ({len(x)} points)')
            
            plt.title(f"{file_name} | Label: {label_val} | {band_name} ({freq_desc}) | n,n+{offset}")
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

print("✅ 주파수 대역 기반 세그먼트별 시각화 완료!")
print("   파일 저장 규칙:")
print("   - _S1_band_seg##.png: S1 주파수 대역 2차 세그먼트 (segment_size > 0인 경우)")
print("   - _S2_band_seg##.png: S2 주파수 대역 2차 세그먼트 (segment_size > 0인 경우)")
print("   - _Clean_band_seg##.png: 잡음 제거 대역 2차 세그먼트 (segment_size > 0인 경우)")
print("   - _S1_band.png: S1 주파수 대역 전체 (segment_size = 0인 경우)")
print("   - _S2_band.png: S2 주파수 대역 전체 (segment_size = 0인 경우)")
print("   - _Clean_band.png: 잡음 제거 대역 전체 (segment_size = 0인 경우)")
print(f"   주파수 대역: S1({s1_freq_low}-{s1_freq_high}Hz), S2({s2_freq_low}-{s2_freq_high}Hz), Clean({noise_freq_high}Hz+)")