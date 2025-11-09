import os
import numpy as np
import librosa
import pywt
import pandas as pd
from scipy.signal import butter, lfilter, find_peaks
from scipy.io import wavfile # 세그먼트 저장을 위해 추가

# --- Configuration ---
BASE_PROJECT_DIR = r'/workspace'
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, 'wavelet_v11/train') # 저장 경로 변경
TRAINING_SETS = ['training']

# Wavelet configuration
WAVELET_FAMILY = 'db4'
DECOMPOSITION_LEVELS = 5

# --- NEW: Segmentation Configuration ---
# 심박수 소리는 주로 25-400Hz 범위에 있습니다.
LOWCUT = 25.0
HIGHCUT = 400.0
SAMPLE_RATE = 2000 # 원본 데이터 샘플링 레이트

# ===================================================================
# NEW FUNCTION: 오디오 세그먼트 분할
# ===================================================================
def segment_heartbeat_audio(audio_data, sr):
    """
    오디오 데이터를 심박 주기(세그먼트)별로 분할합니다.
    피크 감지 기법을 사용하여 각 심박의 시작점을 찾습니다.
    
    Args:
        audio_data (np.array): 오디오 데이터
        sr (int): 샘플링 레이트

    Returns:
        list of np.array: 각 심박 주기에 해당하는 오디오 세그먼트 리스트
    """
    # 1. Band-pass filter: 노이즈를 제거하고 심박 소리만 남깁니다.
    nyquist = 0.5 * sr
    low = LOWCUT / nyquist
    high = HIGHCUT / nyquist
    b, a = butter(5, [low, high], btype='band')
    filtered_audio = lfilter(b, a, audio_data)
    
    # 2. 진폭 엔벨로프 생성
    # 신호 정류 (절댓값)
    rectified_audio = np.abs(filtered_audio)
    
    # 저역 통과 필터로 부드러운 엔벨로프 생성
    # 8Hz cutoff은 심박 봉투선을 그리기에 적합합니다.
    b_lp, a_lp = butter(5, 8.0/nyquist, btype='low')
    envelope = lfilter(b_lp, a_lp, rectified_audio)

    # 3. 피크(Peak) 지점 찾기
    # 심박은 보통 1초에 2번(120bpm)을 넘기 힘드므로, 최소 0.3초 간격을 둡니다.
    min_distance = int(0.3 * sr)
    peaks, _ = find_peaks(envelope, height=np.mean(envelope), distance=min_distance)
    
    if len(peaks) < 2:
        return [] # 세그먼트를 나눌 수 없으면 빈 리스트 반환

    # 4. 피크를 기준으로 오디오 세그먼트 생성
    segments = []
    for i in range(len(peaks) - 1):
        start_sample = peaks[i]
        end_sample = peaks[i+1]
        segments.append(audio_data[start_sample:end_sample])
        
    return segments

# ===================================================================
# MODIFIED FUNCTION: 파일 경로 대신 오디오 '세그먼트'를 입력받도록 수정
# ===================================================================
def extract_features_from_segment(segment):
    """
    하나의 오디오 '세그먼트'에서 30개의 웨이블릿 특징을 추출합니다.
    """
    if len(segment) == 0:

        return None
    TARGET_LENGTH = 512  # 원하는 길이

    # 세그먼트 길이 맞추기 (패딩 또는 자르기)
    if len(segment) < TARGET_LENGTH:
        segment = np.pad(segment, (0, TARGET_LENGTH - len(segment)), mode='constant')
    elif len(segment) > TARGET_LENGTH:
        segment = segment[:TARGET_LENGTH]

    try:
        # Perform 5-level wavelet decomposition
        coeffs = pywt.wavedec(segment, WAVELET_FAMILY, level=DECOMPOSITION_LEVELS)
        
        features = []
        level_means = []
        
        for level_coeffs in coeffs:
            mean_abs = np.mean(np.abs(level_coeffs))
            level_means.append(mean_abs)
        
        for i, level_coeffs in enumerate(coeffs):
            mean_abs = np.mean(np.abs(level_coeffs))
            mean_square = np.mean(np.square(level_coeffs))
            std_dev = np.std(level_coeffs)
            
            if i == 0:
                ratio = abs(level_means[0] / level_means[1]) if len(level_means) > 1 and level_means[1] != 0 else 0.0
            elif i == len(coeffs) - 1:
                ratio = abs(level_means[i] / level_means[i-1]) if level_means[i-1] != 0 else 0.0
            else:
                ratio = abs(level_means[i] / level_means[i+1]) if level_means[i+1] != 0 else 0.0

            median_val = np.median(level_coeffs)
            features.extend([mean_abs, mean_square, std_dev, ratio, median_val])
        
        return np.array(features[:30])
        
    except Exception as e:
        print(f"Error processing a segment: {e}")
        return None

def load_reference_labels(data_dir):
    reference_file = os.path.join(data_dir, 'REFERENCE.csv')
    labels = {}
    if os.path.exists(reference_file):
        try:
            with open(reference_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                        filename, label = line.split(',')
                        labels[filename] = 0 if int(label) == -1 else 1
        except Exception as e:
            print(f"Error reading {reference_file}: {e}")
    return labels

def create_feature_column_names():
    columns = []
    level_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
    for level_name in level_names:
        columns.extend([
            f'{level_name}_mean_abs', f'{level_name}_mean_square', 
            f'{level_name}_std', f'{level_name}_ratio', f'{level_name}_median'
        ])
    return columns

# ===================================================================
# MODIFIED FUNCTION: 메인 로직 수정
# ===================================================================
def process_all_training_sets():
    """
    모든 훈련 세트의 오디오 파일을 세그먼트로 나누고, 
    각 세그먼트에서 특징을 추출하여 CSV 파일로 저장합니다.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    feature_columns = create_feature_column_names()
    # 'segment_id' 컬럼 추가하여 몇 번째 세그먼트인지 추적
    all_columns = feature_columns + ['label']
    
    all_data = []
    
    print("Starting wavelet feature extraction from AUDIO SEGMENTS...")
    
    for set_name in TRAINING_SETS:
        print(f"\n--- Processing {set_name} ---")
        
        data_dir = os.path.join(BASE_PROJECT_DIR, set_name)
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found, skipping...")
            continue
            
        reference_labels = load_reference_labels(data_dir)
        if not reference_labels:
            print(f"No reference labels found for {set_name}, skipping...")
            continue
            
        print(f"Found {len(reference_labels)} reference labels")
        
        processed_files = 0
        total_segments = 0
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.wav'):
                file_id = os.path.splitext(filename)[0]
                
                if file_id not in reference_labels:
                    continue
                
                audio_path = os.path.join(data_dir, filename)
                
                # 1. 오디오 파일 불러오기
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                
                # 2. 오디오를 세그먼트로 분할
                segments = segment_heartbeat_audio(y, sr)
                
                if not segments:
                    print(f"Could not find any segments in {filename}, skipping...")
                    continue

                processed_files += 1
                
                # 3. 각 세그먼트에서 특징 추출
                for i, segment in enumerate(segments):
                    features = extract_features_from_segment(segment)
                    
                    if features is not None and len(features) == 30:
                        row_data = list(features) + [reference_labels[file_id]]
                            # file_id,         # 원본 파일명
                            # i,               # 세그먼트 번호
                            # set_name,        # 세트명
                            
                        
                        all_data.append(row_data)
                        total_segments += 1
                
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files} files from {set_name} (Total segments: {total_segments})")

        print(f"Completed {set_name}: {processed_files} files processed, {total_segments} segments created.")

    if all_data:
        print("\n--- Saving results ---")
        df = pd.DataFrame(all_data, columns=all_columns)
        
        csv_path = os.path.join(SAVE_DIR, 'wavelet_30_features_segmented_complete.csv')
        df.to_csv(csv_path, index=False)
        print(f"Complete segmented dataset saved to: {csv_path}")
        print(f"\n--- Dataset Statistics ---")
        print(f"Total segments (samples): {len(df)}")
        print(f"Features per segment: {len(feature_columns)}")
        print(f"Label distribution:")
        print(df['label'].value_counts().sort_index())
    
    else:
        print("No data was processed!")

if __name__ == "__main__":
    process_all_training_sets()