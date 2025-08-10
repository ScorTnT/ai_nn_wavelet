import os
import numpy as np
import librosa
import pywt
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.spatial.distance import pdist

# --- Configuration ---
# Base directory where training sets are located (absolute path in the container)
BASE_PROJECT_DIR = r'/workspace'
# Directory to save the processed wavelet data (CSV format)
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, 'wavelet_v8')
# Training set directory
TRAINING_DIR = 'validation'
csv_name = 'Test60.csv'
feature_name = 'feature_description_60.txt'
# Wavelet configuration
WAVELET_FAMILY = 'db4'
DECOMPOSITION_LEVELS = 5

def load_reference_labels(data_dir):
    """
    Load reference labels from REFERENCE.csv file.
    Returns a dictionary mapping filename to label.
    """
    reference_file = os.path.join(data_dir, 'REFERENCE.csv')
    labels = {}
    
    if os.path.exists(reference_file):
        try:
            # REFERENCE.csv 형식이 헤더 없이 파일명,라벨 형태라고 가정
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

def extract_60_wavelet_features(audio_path):
    """
    Extract 60 wavelet features from audio file using 5-level DWT decomposition.
    
    5-level decomposition creates 6 scale levels (1 approximation + 5 detail levels).
    For each of the 6 levels, extract 10 features:
    1. RMS (Root Mean Square)
    2. 왜도 (Skewness)
    3. 첨도 (Kurtosis)
    4. 평균 절대 편차 (Mean Absolute Deviation)
    5. 사분위 범위 (Interquartile Range)
    6. 중앙값 (Median)
    7. MSQ (Mean Square)
    8. 엔트로피 (Entropy)
    9. 로그 편차의 평균 (Mean of Log Deviations)
    10. 절댓값 평균 (Mean Absolute Value)
    
    Order: cD1, cD2, cD3, cD4, cD5, cA5
    
    Returns:
        numpy array of 60 features (6 levels × 10 features = 60)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=2000)
        
        # Perform 5-level wavelet decomposition
        # This creates 6 levels: 1 approximation (cA5) + 5 detail levels (cD5, cD4, cD3, cD2, cD1)
        coeffs = pywt.wavedec(y, WAVELET_FAMILY, level=DECOMPOSITION_LEVELS)
        
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
        
        # Extract 10 features for each of the 6 levels
        for level_coeffs in reordered_coeffs:
            level_coeffs = np.array(level_coeffs)
            
            # 1. RMS (Root Mean Square)
            rms = np.sqrt(np.mean(np.square(level_coeffs)))
            
            # 2. 왜도 (Skewness)
            skewness = skew(level_coeffs)
            
            # 3. 첨도 (Kurtosis)
            kurt = kurtosis(level_coeffs)
            
            # 4. 평균 절대 편차 (Mean Absolute Deviation)
            mean_val = np.mean(level_coeffs)
            mad = np.mean(np.abs(level_coeffs - mean_val))
            
            # 5. 사분위 범위 (Interquartile Range)
            q75, q25 = np.percentile(level_coeffs, [75, 25])
            iqr = q75 - q25
            
            # 6. 중앙값 (Median)
            median_val = np.median(level_coeffs)
            
            # 7. MSQ (Mean Square)
            msq = np.mean(np.square(level_coeffs))
            
            # 8. 엔트로피 (Entropy)
            # Normalize coefficients for entropy calculation
            normalized_coeffs = np.abs(level_coeffs)
            if np.sum(normalized_coeffs) > 0:
                normalized_coeffs = normalized_coeffs / np.sum(normalized_coeffs)
                # Add small epsilon to avoid log(0)
                normalized_coeffs = normalized_coeffs + 1e-12
                entropy_val = -np.sum(normalized_coeffs * np.log2(normalized_coeffs))
            else:
                entropy_val = 0.0
            
            # 9. 로그 편차의 평균 (Mean of Log Deviations)
            abs_coeffs = np.abs(level_coeffs)
            # Add small epsilon to avoid log(0)
            abs_coeffs = abs_coeffs + 1e-12
            log_dev_mean = np.mean(np.log(abs_coeffs))
            
            # 10. 절댓값 평균 (Mean Absolute Value)
            mean_abs = np.mean(np.abs(level_coeffs))
            
            # Add 10 features for this level
            features.extend([
                rms, skewness, kurt, mad, iqr, 
                median_val, msq, entropy_val, log_dev_mean, mean_abs
            ])
        
        return np.array(features[:60])  # Ensure exactly 60 features
        
    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        return None

def create_feature_column_names():
    """
    Create descriptive column names for the 60 wavelet features.
    6 levels × 10 features = 60 features total
    Order: cD1, cD2, cD3, cD4, cD5, cA5
    """
    columns = []
    
    # 6 levels in new order: detail levels (low to high) + approximation
    level_names = ['cD1', 'cD2', 'cD3', 'cD4', 'cD5', 'cA5']
    
    # 10 feature names according to readme.md
    feature_names = [
        'rms',           # 1. RMS
        'skewness',      # 2. 왜도
        'kurtosis',      # 3. 첨도
        'mad',           # 4. 평균 절대 편차
        'iqr',           # 5. 사분위 범위
        'median',        # 6. 중앙값
        'msq',           # 7. MSQ
        'entropy',       # 8. 엔트로피
        'log_dev_mean',  # 9. 로그 편차의 평균
        'mean_abs'       # 10. 절댓값 평균
    ]
    
    for level_name in level_names:
        for feature_name in feature_names:
            columns.append(f'{level_name}_{feature_name}')
    
    return columns

def process_training_dataset():
    """
    Process training dataset and save features to CSV files.
    """
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Column names for CSV (filename 제거)
    feature_columns = create_feature_column_names()
    all_columns = feature_columns + ['label']
    
    # Initialize lists to store all data
    all_data = []
    
    print("Starting 60-feature wavelet feature extraction...")
    print(f"논문 기준: 6개 레벨 × 10가지 특징 = 60개 특징")
    print(f"10 Features per level: RMS, 왜도, 첨도, 평균절대편차, 사분위범위, 중앙값, MSQ, 엔트로피, 로그편차평균, 절댓값평균")
    print(f"Level order: cD1, cD2, cD3, cD4, cD5, cA5")
    print(f"Total features: {len(feature_columns)}")
    
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
    processed_count = 0
    skipped_count = 0
    
    for filename in sorted(wav_files):
        file_id = os.path.splitext(filename)[0]
        
        # Check if we have a label for this file
        if file_id not in reference_labels:
            print(f"No label found for {file_id}, skipping...")
            skipped_count += 1
            continue
        
        audio_path = os.path.join(data_dir, filename)
        
        # Extract 60 wavelet features
        features = extract_60_wavelet_features(audio_path)
        
        if features is not None and len(features) == 60:
            # Create data row (filename 제거, label만 추가)
            row_data = list(features) + [reference_labels[file_id]]
            all_data.append(row_data)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
        else:
            print(f"Failed to extract features from {filename}")
            skipped_count += 1
    
    print(f"Processing completed: {processed_count} files processed, {skipped_count} files skipped")
    
    # Create DataFrame and save to CSV
    if all_data:
        print(f"\n--- Saving results ---")
        df = pd.DataFrame(all_data, columns=all_columns)
        
        # Save complete dataset
        csv_path = os.path.join(SAVE_DIR, csv_name)
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to: {csv_path}")
        
        # Print dataset statistics
        print(f"\n--- Dataset Statistics ---")
        print(f"Total samples: {len(df)}")
        print(f"Features per sample: {len(feature_columns)}")
        print(f"Label distribution:")
        label_dist = df['label'].value_counts().sort_index()
        for label, count in label_dist.items():
            label_type = "정상" if label == 0 else "비정상"
            print(f"  {label} ({label_type}): {count}")
        
        # Check for any missing values
        print(f"\nMissing values check:")
        missing_count = df.isnull().sum().sum()
        print(f"Total missing values: {missing_count}")
        
        if missing_count > 0:
            print("Columns with missing values:")
            for col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    print(f"  {col}: {missing}")
        
        # Save feature description
        feature_desc_path = os.path.join(SAVE_DIR, feature_name)
        with open(feature_desc_path, 'w', encoding='utf-8') as f:
            f.write("60 Wavelet Features Description\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Source: {TRAINING_DIR} dataset\n")
            f.write("Wavelet: Daubechies 4 (db4)\n")
            f.write("Decomposition levels: 5 (creates 6 scale levels)\n")
            f.write("Total features: 60 (6 levels × 10 features)\n")
            f.write("Level order: cD1, cD2, cD3, cD4, cD5, cA5\n\n")
            f.write("5-level discrete wavelet transform creates 6 scale levels:\n")
            f.write("- cD1, cD2, cD3, cD4, cD5: Detail coefficients (levels 1-5)\n")
            f.write("- cA5: Approximation coefficients (level 5)\n\n")
            f.write("For each of the 6 levels, extract 10 features:\n")
            f.write("1. RMS (Root Mean Square)\n")
            f.write("2. 왜도 (Skewness)\n")
            f.write("3. 첨도 (Kurtosis)\n")
            f.write("4. 평균 절대 편차 (Mean Absolute Deviation)\n")
            f.write("5. 사분위 범위 (Interquartile Range)\n")
            f.write("6. 중앙값 (Median)\n")
            f.write("7. MSQ (Mean Square)\n")
            f.write("8. 엔트로피 (Entropy)\n")
            f.write("9. 로그 편차의 평균 (Mean of Log Deviations)\n")
            f.write("10. 절댓값 평균 (Mean Absolute Value)\n\n")
            f.write("Column names:\n")
            for i, col in enumerate(feature_columns, 1):
                f.write(f"{i:2d}. {col}\n")
            
            f.write(f"\nDataset Summary:\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Label distribution:\n")
            for label, count in df['label'].value_counts().sort_index().items():
                label_type = "정상" if label == 0 else "비정상"
                f.write(f"  Label {label} ({label_type}): {count} samples\n")
        
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
            result_path = process_training_dataset()
            if result_path:
                print(f"\n✅ 60-feature wavelet extraction completed successfully!")
                print(f"Results saved to: {result_path}")
            else:
                print(f"\n❌ Feature extraction failed!")