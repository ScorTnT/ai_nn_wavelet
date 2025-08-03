import os
import numpy as np
import librosa
import pywt
import pandas as pd
from scipy.stats import skew, kurtosis

# --- Configuration ---
# Base directory where training sets are located
BASE_PROJECT_DIR = r'/workspace'
# Directory to save the processed wavelet data (CSV format)
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, 'wavelet_v7_features')
# List of training set subdirectories
TRAINING_SETS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

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
            with open(reference_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                        filename, label = line.split(',')
                        # Convert PhysioNet labels: -1 (normal) -> 0, 1 (abnormal) -> 1
                        labels[filename] = 0 if int(label) == -1 else 1
        except Exception as e:
            print(f"Error reading {reference_file}: {e}")
    
    return labels

def extract_30_wavelet_features(audio_path):
    """
    Extract 30 wavelet features from audio file using 5-level DWT decomposition.
    
    5-level decomposition creates 6 scale levels (1 approximation + 5 detail levels).
    For each of the 6 levels, extract 5 features:
    1. Mean of absolute values of all coefficients
    2. Mean of squared values of all coefficients  
    3. Standard deviation of all coefficients
    4. Ratio of absolute mean values between adjacent levels
    5. Median of all coefficients
    
    Returns:
        numpy array of 30 features (6 levels × 5 features = 30)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=2000)
        
        # Perform 5-level wavelet decomposition
        # This creates 6 levels: 1 approximation (cA5) + 5 detail levels (cD5, cD4, cD3, cD2, cD1)
        coeffs = pywt.wavedec(y, WAVELET_FAMILY, level=DECOMPOSITION_LEVELS)
        
        features = []
        level_means = []  # Store means for ratio calculations
        
        # First pass: calculate means for ratio calculations
        for level_coeffs in coeffs:
            mean_abs = np.mean(np.abs(level_coeffs))
            level_means.append(mean_abs)
        
        # Second pass: extract 5 features for each of the 6 levels
        for i, level_coeffs in enumerate(coeffs):
            # 1. Mean of absolute values of all coefficients
            mean_abs = np.mean(np.abs(level_coeffs))
            
            # 2. Mean of squared values of all coefficients
            mean_square = np.mean(np.square(level_coeffs))
            
            # 3. Standard deviation of all coefficients
            std_dev = np.std(level_coeffs)
            
            # 4. Ratio of absolute mean values between adjacent levels
            if i == 0:
                # For first level (approximation), use ratio with next level
                if len(level_means) > 1 and level_means[1] != 0:
                    ratio = abs(level_means[0] / level_means[1])
                else:
                    ratio = 0.0
            elif i == len(coeffs) - 1:
                # For last level, use ratio with previous level
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
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        return None

def create_feature_column_names():
    """
    Create descriptive column names for the 30 wavelet features.
    6 levels × 5 features = 30 features total
    """
    columns = []
    
    # 6 levels from 5-level decomposition: approximation + 5 detail levels
    level_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
    
    for level_name in level_names:
        columns.extend([
            f'{level_name}_mean_abs',      # 1. Mean of absolute values
            f'{level_name}_mean_square',   # 2. Mean of squared values
            f'{level_name}_std',           # 3. Standard deviation
            f'{level_name}_ratio',         # 4. Ratio with adjacent level
            f'{level_name}_median'         # 5. Median
        ])
    
    return columns

def process_all_training_sets():
    """
    Process all training sets and save features to CSV files.
    """
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Column names for CSV
    feature_columns = create_feature_column_names()
    all_columns = feature_columns + ['label', 'filename', 'set_name']
    
    # Initialize lists to store all data
    all_data = []
    
    print("Starting wavelet feature extraction with 30 features...")
    print(f"논문 2.2 웨이블릿 변환 기준: 6개 레벨 × 5가지 특징 = 30개 특징")
    print(f"Features per level: Mean_abs, Mean_square, Std, Ratio, Median")
    print(f"Total features: {len(feature_columns)}")
    
    for set_name in TRAINING_SETS:
        print(f"\n--- Processing {set_name} ---")
        
        data_dir = os.path.join(BASE_PROJECT_DIR, set_name)
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found, skipping...")
            continue
        
        # Load reference labels
        reference_labels = load_reference_labels(data_dir)
        if not reference_labels:
            print(f"No reference labels found for {set_name}, skipping...")
            continue
        
        print(f"Found {len(reference_labels)} reference labels")
        
        # Process each audio file
        processed_count = 0
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.wav'):
                file_id = os.path.splitext(filename)[0]
                
                # Check if we have a label for this file
                if file_id not in reference_labels:
                    print(f"No label found for {file_id}, skipping...")
                    continue
                
                audio_path = os.path.join(data_dir, filename)
                
                # Extract 30 wavelet features
                features = extract_30_wavelet_features(audio_path)
                
                if features is not None and len(features) == 30:
                    # Create data row
                    row_data = list(features) + [
                        reference_labels[file_id],  # label
                        file_id,                    # filename
                        set_name                    # set_name
                    ]
                    all_data.append(row_data)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Processed {processed_count} files from {set_name}")
        
        print(f"Completed {set_name}: {processed_count} files processed")
    
    # Create DataFrame and save to CSV
    if all_data:
        print(f"\n--- Saving results ---")
        df = pd.DataFrame(all_data, columns=all_columns)
        
        # Save complete dataset
        csv_path = os.path.join(SAVE_DIR, 'wavelet_30_features_complete.csv')
        df.to_csv(csv_path, index=False)
        print(f"Complete dataset saved to: {csv_path}")
        
        # Save separate files for each training set
        for set_name in TRAINING_SETS:
            set_data = df[df['set_name'] == set_name]
            if not set_data.empty:
                set_csv_path = os.path.join(SAVE_DIR, f'wavelet_30_features_{set_name}.csv')
                set_data.to_csv(set_csv_path, index=False)
                print(f"{set_name} dataset saved to: {set_csv_path}")
        
        # Print dataset statistics
        print(f"\n--- Dataset Statistics ---")
        print(f"Total samples: {len(df)}")
        print(f"Features per sample: {len(feature_columns)}")
        print(f"Label distribution:")
        print(df['label'].value_counts().sort_index())
        print(f"\nSet distribution:")
        print(df['set_name'].value_counts())
        
        # Save feature description
        feature_desc_path = os.path.join(SAVE_DIR, 'feature_description.txt')
        with open(feature_desc_path, 'w', encoding='utf-8') as f:
            f.write("30 Wavelet Features Description (논문 2.2 웨이블릿 변환 기준)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Wavelet: Daubechies 4 (db4)\n")
            f.write("Decomposition levels: 5 (creates 6 scale levels)\n")
            f.write("Total features: 30 (6 levels × 5 features)\n\n")
            f.write("5-level discrete wavelet transform creates 6 scale levels:\n")
            f.write("- cA5: Approximation coefficients (level 5)\n")
            f.write("- cD5, cD4, cD3, cD2, cD1: Detail coefficients (levels 5-1)\n\n")
            f.write("For each of the 6 levels, extract 5 features:\n")
            f.write("1. Mean of absolute values of all coefficients in the level\n")
            f.write("2. Mean of squared values of all coefficients in the level\n")
            f.write("3. Standard deviation of all coefficients in the level\n")
            f.write("4. Ratio of absolute mean values between adjacent levels\n")
            f.write("5. Median of all coefficients in the level\n\n")
            f.write("Column names:\n")
            for i, col in enumerate(feature_columns, 1):
                f.write(f"{i:2d}. {col}\n")
        
        print(f"Feature description saved to: {feature_desc_path}")
        
    else:
        print("No data was processed!")

if __name__ == "__main__":
    process_all_training_sets()
