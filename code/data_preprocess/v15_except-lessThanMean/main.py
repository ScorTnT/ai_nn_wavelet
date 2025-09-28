import os
import numpy as np
import librosa
import pywt
import pandas as pd
from scipy.stats import skew, kurtosis

# --- Configuration ---
BASE_PROJECT_DIR = r'/workspace'
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, 'wavelet_v15/test')
TRAINING_DIR = 'validation'  # 'training' 'validation'

# Wavelet configuration
WAVELET_FAMILY = 'db4'
DECOMPOSITION_LEVELS = 5

# Segmentation configuration
SEGMENT_LENGTH = 1024  # 256, 512, 1024 등으로 변경 가능
SAMPLE_RATE = 2000

def load_reference_labels(data_dir):
    """
    Load reference labels from REFERENCE.csv file.
    Returns a dictionary mapping filename to label.
    """
    reference_file = os.path.join(data_dir, 'REFERENCE.csv')
    labels = {}
    
    if os.path.exists(reference_file):
        try:
            with open(reference_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            file_id = parts[0]
                            label = int(parts[1])
                            # -1 (정상) -> 0, 1 (비정상) -> 1로 변환
                            binary_label = 0 if label == -1 else 1
                            labels[file_id] = binary_label
        except Exception as e:
            print(f"Error reading {reference_file}: {e}")
    
    return labels

def segment_audio_by_length(audio_data, segment_length):
    """
    오디오 데이터를 고정 길이 세그먼트로 분할
    
    Args:
        audio_data: 1D numpy array
        segment_length: 세그먼트 길이 (샘플 수)
    
    Returns:
        list of segments
    """
    segments = []
    audio_length = len(audio_data)
    
    # 전체 가능한 세그먼트 수 계산
    num_segments = audio_length // segment_length
    
    # 세그먼트 추출
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = audio_data[start_idx:end_idx]
        segments.append(segment)
    
    return segments

def extract_30_wavelet_features(segment):
    """
    하나의 세그먼트에서 30개 웨이블릿 특징 추출
    
    Returns:
        numpy array of 30 features (6 levels × 5 features = 30)
    """
    try:
        # Perform 5-level wavelet decomposition
        coeffs = pywt.wavedec(segment, WAVELET_FAMILY, level=DECOMPOSITION_LEVELS)
        
        # Reorder coefficients: from [cA5, cD5, cD4, cD3, cD2, cD1] to [cD1, cD2, cD3, cD4, cD5, cA5]
        reordered_coeffs = [
            coeffs[5],  # cD1
            coeffs[4],  # cD2  
            coeffs[3],  # cD3
            coeffs[2],  # cD4
            coeffs[1],  # cD5
            coeffs[0]   # cA5
        ]
        
        features = []
        level_means = []  # Store means for ratio calculations
        
        # First pass: calculate means for ratio calculations
        for level_coeffs in reordered_coeffs:
            mean_abs = np.mean(np.abs(level_coeffs))
            level_means.append(mean_abs)
        
        # Second pass: extract 5 features for each of the 6 levels
        for i, level_coeffs in enumerate(reordered_coeffs):
            # 1. Mean of absolute values of all coefficients
            mean_abs = np.mean(np.abs(level_coeffs))
            
            # 2. Mean of squared values of all coefficients
            mean_square = np.mean(np.square(level_coeffs))
            
            # 3. Standard deviation of all coefficients
            std_dev = np.std(level_coeffs)
            
            # 4. Ratio of absolute mean values between adjacent levels
            if i == 0:
                # For first level (cD1), use ratio with next level (cD2)
                if len(level_means) > 1 and level_means[1] != 0:
                    ratio = abs(level_means[0] / level_means[1])
                else:
                    ratio = 0.0
            elif i == len(reordered_coeffs) - 1:
                # For last level (cA5), use ratio with previous level (cD5)
                if level_means[i-1] != 0:
                    ratio = abs(level_means[i] / level_means[i-1])
                else:
                    ratio = 0.0
            else:
                # For middle levels, use ratio with next level
                if level_means[i+1] != 0:
                    ratio = abs(level_means[i] / level_means[i+1])
                else:
                    ratio = 0.0
            
            # 5. Median of all coefficients
            median_val = np.median(level_coeffs)
            
            # Add 5 features for this level
            features.extend([mean_abs, mean_square, std_dev, ratio, median_val])
        
        return np.array(features[:30])  # Ensure exactly 30 features
        
    except Exception as e:
        print(f"Error extracting features from segment: {e}")
        return None

def filter_segments_by_individual_feature_means(segments_features):
    """
    각 특징별로 개별적으로 평균을 넘는 세그먼트들을 선택하고 중복 제거
    
    Args:
        segments_features: list of numpy arrays, 각 세그먼트의 30개 특징값들
    
    Returns:
        filtered_features: list of numpy arrays, 필터링된 세그먼트들의 특징값들
    """
    if not segments_features:
        return []
    
    # Convert to numpy array for easier calculation
    features_array = np.array(segments_features)  # shape: (num_segments, 30)
    num_segments, num_features = features_array.shape
    
    print(f"    원본 세그먼트 수: {num_segments}")
    print(f"    특징별 개별 필터링 시작 (30개 특징)")
    
    # Set to store indices of segments that pass any feature filter
    selected_indices = set()
    
    # For each feature, find segments that are >= mean for that feature
    for feature_idx in range(num_features):
        feature_values = features_array[:, feature_idx]
        feature_mean = np.mean(feature_values)
        
        # Find segments where this feature is >= mean
        above_mean_indices = np.where(feature_values >= feature_mean)[0]
        
        # Add these indices to our selected set
        selected_indices.update(above_mean_indices)
        
        print(f"      특징 {feature_idx+1}: 평균={feature_mean:.6f}, "
              f"평균 이상 세그먼트={len(above_mean_indices)}개")
    
    # Convert set to sorted list
    selected_indices = sorted(list(selected_indices))
    
    # Extract the selected segments
    filtered_features = [segments_features[i] for i in selected_indices]
    
    filter_ratio = len(filtered_features) / num_segments * 100
    print(f"    중복 제거 후 최종 선택된 세그먼트: {len(filtered_features)}개 ({filter_ratio:.1f}% 유지)")
    
    return filtered_features

def create_feature_column_names():
    """
    Create descriptive column names for the 30 wavelet features.
    6 levels × 5 features = 30 features total
    Order: cD1, cD2, cD3, cD4, cD5, cA5
    """
    columns = []
    
    # 6 levels in new order: detail levels (low to high) + approximation
    level_names = ['cD1', 'cD2', 'cD3', 'cD4', 'cD5', 'cA5']
    
    for level_name in level_names:
        columns.extend([
            f'{level_name}_mean_abs',      # 1. Mean of absolute values
            f'{level_name}_mean_square',   # 2. Mean of squared values
            f'{level_name}_std',           # 3. Standard deviation
            f'{level_name}_ratio',         # 4. Ratio with adjacent level
            f'{level_name}_median'         # 5. Median
        ])
    
    return columns

def process_heartbeat_dataset():
    """
    Process heartbeat dataset with individual feature-based filtering and save features to CSV files.
    """
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Column names for CSV
    feature_columns = create_feature_column_names()
    all_columns = feature_columns + ['label']
    
    # Initialize lists to store all data
    all_data = []
    processed_files_list = []  # 파일명 추적용
    
    print("=== 심장박동 오디오 세그먼트 웨이블릿 특징 추출 (개별 특징별 평균 필터링) ===")
    print(f"세그먼트 길이: {SEGMENT_LENGTH} 샘플 ({SEGMENT_LENGTH/SAMPLE_RATE:.3f}초)")
    print(f"웨이블릿: {WAVELET_FAMILY}, 분해 레벨: {DECOMPOSITION_LEVELS}")
    print(f"특징 개수: {len(feature_columns)} (6 레벨 × 5 특징)")
    print(f"필터링 방식: 각 특징별로 개별적으로 평균 이상인 세그먼트 선택 후 중복 제거")
    print(f"Level order: cD1, cD2, cD3, cD4, cD5, cA5")
    
    data_dir = os.path.join(BASE_PROJECT_DIR, TRAINING_DIR)
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return
    
    print(f"\n--- Processing {TRAINING_DIR} ---")
    
    # Load reference labels
    reference_labels = load_reference_labels(data_dir)
    if not reference_labels:
        print(f"No reference labels found for {TRAINING_DIR}")
        return
    
    print(f"Found {len(reference_labels)} reference labels")
    
    # 라벨 분포 확인
    label_counts = {}
    for label in reference_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"라벨 분포:")
    for label, count in sorted(label_counts.items()):
        label_type = "정상" if label == 0 else "비정상"
        print(f"  {label} ({label_type}): {count}개")
    
    # Get all wav files
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files")
    
    # Process each audio file
    processed_files = 0
    skipped_files = 0
    total_segments_before = 0
    total_segments_after = 0
    
    for filename in sorted(wav_files):
        file_id = os.path.splitext(filename)[0]
        
        # Check if we have a label for this file
        if file_id not in reference_labels:
            print(f"No label found for {file_id}, skipping...")
            skipped_files += 1
            continue
        
        audio_path = os.path.join(data_dir, filename)
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            print(f"\n처리 중: {filename}")
            
            # Segment audio by fixed length
            segments = segment_audio_by_length(y, SEGMENT_LENGTH)
            
            if not segments:
                print(f"  세그먼트를 추출할 수 없습니다: {filename}")
                skipped_files += 1
                continue
            
            # Extract features from each segment
            segments_features = []
            for seg_idx, segment in enumerate(segments):
                features = extract_30_wavelet_features(segment)
                
                if features is not None and len(features) == 30:
                    segments_features.append(features)
                else:
                    print(f"  세그먼트 {seg_idx}에서 특징 추출 실패")
            
            total_segments_before += len(segments_features)
            
            if not segments_features:
                print(f"  특징을 추출한 세그먼트가 없습니다: {filename}")
                skipped_files += 1
                continue
            
            # Apply individual feature-based filtering
            filtered_features = filter_segments_by_individual_feature_means(segments_features)
            total_segments_after += len(filtered_features)
            
            if not filtered_features:
                print(f"  필터링 후 남은 세그먼트가 없습니다: {filename}")
                skipped_files += 1
                continue
            
            # Add filtered segments to dataset
            for features in filtered_features:
                row_data = list(features) + [
                    reference_labels[file_id]  # label
                ]
                all_data.append(row_data)
            
            processed_files_list.append(file_id)
            print(f"  최종 사용된 세그먼트: {len(filtered_features)}개")
            processed_files += 1
            
            if processed_files % 10 == 0:
                current_filter_ratio = (total_segments_after / total_segments_before * 100) if total_segments_before > 0 else 0
                print(f"\n진행 상황: {processed_files}/{len(wav_files)} 파일 처리 완료")
                print(f"  전체 필터링 비율: {total_segments_after}/{total_segments_before} ({current_filter_ratio:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_files += 1
    
    final_filter_ratio = (total_segments_after / total_segments_before * 100) if total_segments_before > 0 else 0
    
    print(f"\n=== 처리 완료 ===")
    print(f"처리된 파일: {processed_files}개")
    print(f"건너뛴 파일: {skipped_files}개")
    print(f"필터링 전 총 세그먼트: {total_segments_before}개")
    print(f"필터링 후 총 세그먼트: {total_segments_after}개 ({final_filter_ratio:.1f}% 유지)")
    
    # Create DataFrame and save to CSV
    if all_data:
        print(f"\n--- 결과 저장 ---")
        df = pd.DataFrame(all_data, columns=all_columns)
        
        # Save complete dataset
        csv_filename = f'wavelet_30_features_individual_filtered_{SEGMENT_LENGTH}_{TRAINING_DIR}.csv'
        csv_path = os.path.join(SAVE_DIR, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to: {csv_path}")
        
        # Print dataset statistics
        print(f"\n--- Dataset Statistics ---")
        print(f"Total segments (after individual filtering): {len(df)}")
        print(f"Features per segment: {len(feature_columns)}")
        print(f"Unique files processed: {len(processed_files_list)}")
        print(f"Segments per file (avg): {len(df) / len(processed_files_list):.1f}")
        
        print(f"\nLabel distribution:")
        label_dist = df['label'].value_counts().sort_index()
        for label, count in label_dist.items():
            label_type = "정상" if label == 0 else "비정상"
            print(f"  {label} ({label_type}): {count} segments")
        
        # Check for any missing values
        print(f"\nMissing values check:")
        missing_count = df.isnull().sum().sum()
        print(f"Total missing values: {missing_count}")
        
        # Save feature description
        desc_filename = f'feature_description_individual_filtered_{SEGMENT_LENGTH}.txt'
        feature_desc_path = os.path.join(SAVE_DIR, desc_filename)
        with open(feature_desc_path, 'w', encoding='utf-8') as f:
            f.write(f"30 Wavelet Features Description (Individual Feature Filtered, Segment Length: {SEGMENT_LENGTH})\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Source: {TRAINING_DIR} dataset\n")
            f.write(f"Wavelet: {WAVELET_FAMILY}\n")
            f.write(f"Decomposition levels: {DECOMPOSITION_LEVELS}\n")
            f.write(f"Segment length: {SEGMENT_LENGTH} samples ({SEGMENT_LENGTH/SAMPLE_RATE:.3f} seconds)\n")
            f.write(f"Total features: 30 (6 levels × 5 features)\n")
            f.write("Level order: cD1, cD2, cD3, cD4, cD5, cA5\n\n")
            f.write("Segmentation method: Fixed-length segments with individual feature-based filtering\n")
            f.write("- Audio files are divided into fixed-length segments\n")
            f.write(f"- Each segment is exactly {SEGMENT_LENGTH} samples long\n")
            f.write("- For each file, for each of the 30 features individually:\n")
            f.write("  1. Calculate the mean of that feature across all segments\n")
            f.write("  2. Select segments where that feature >= its mean\n")
            f.write("- Combine all selected segments and remove duplicates\n")
            f.write("- This keeps segments that excel in ANY feature\n\n")
            f.write(f"Filtering Results:\n")
            f.write(f"- Original segments: {total_segments_before}\n")
            f.write(f"- Filtered segments: {total_segments_after}\n")
            f.write(f"- Retention rate: {final_filter_ratio:.1f}%\n\n")
            f.write("For each of the 6 wavelet levels, extract 5 features:\n")
            f.write("1. Mean of absolute values of all coefficients\n")
            f.write("2. Mean of squared values of all coefficients\n")
            f.write("3. Standard deviation of all coefficients\n")
            f.write("4. Ratio of absolute mean values between adjacent levels\n")
            f.write("5. Median of all coefficients\n\n")
            f.write("Column names:\n")
            for i, col in enumerate(feature_columns, 1):
                f.write(f"{i:2d}. {col}\n")
            
            f.write(f"\nDataset Summary:\n")
            f.write(f"Total segments (after individual filtering): {len(df)}\n")
            f.write(f"Unique files processed: {len(processed_files_list)}\n")
            f.write(f"Average segments per file: {len(df) / len(processed_files_list):.1f}\n")
            f.write(f"Label distribution:\n")
            for label, count in df['label'].value_counts().sort_index().items():
                label_type = "정상" if label == 0 else "비정상"
                f.write(f"  Label {label} ({label_type}): {count} segments\n")
        
        print(f"Feature description saved to: {feature_desc_path}")
        
        return csv_path
        
    else:
        print("No data was processed!")
        return None

if __name__ == "__main__":
    # Check if training directory exists
    data_dir = os.path.join(BASE_PROJECT_DIR, TRAINING_DIR)
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        print("Please make sure you have created the training folder first.")
    else:
        # Check if REFERENCE.csv exists
        ref_file = os.path.join(data_dir, 'REFERENCE.csv')
        if not os.path.exists(ref_file):
            print(f"REFERENCE.csv not found in {data_dir}")
            print("Please make sure REFERENCE.csv exists in the training folder.")
        else:
            print(f"Processing {data_dir}...")
            print(f"세그먼트 길이: {SEGMENT_LENGTH} 샘플")
            result_path = process_heartbeat_dataset()
            if result_path:
                print(f"\n✅ 웨이블릿 특징 추출 및 개별 필터링 완료!")
                print(f"결과 파일: {result_path}")
                print(f"\n세그먼트 길이를 변경하려면 SEGMENT_LENGTH 값을 수정하세요.")
                print(f"현재 설정: {SEGMENT_LENGTH} (256, 512, 1024 등으로 변경 가능)")
            else:
                print(f"\n❌ 특징 추출 실패!")