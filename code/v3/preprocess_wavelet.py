import os
import numpy as np
import librosa
import pywt
import pandas as pd

# --- Configuration ---
# Base directory where training sets are located
BASE_PROJECT_DIR = r'/workspace'
# Directory to save the processed wavelet data
SAVE_DIR = os.path.join(BASE_PROJECT_DIR, 'wavelet_v3')
# List of training set subdirectories
TRAINING_SETS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

# Target length for all feature vectors (for consistency)
TARGET_FEATURE_LENGTH = 2000

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

def get_label_from_filename_and_set(filename, set_name, reference_labels):
    """
    Extract label from reference labels dictionary.
    """
    # Remove extension
    file_id = os.path.splitext(filename)[0]
    
    # Get label from reference if available
    if file_id in reference_labels:
        return reference_labels[file_id]
    else:
        print(f"Warning: No label found for {file_id}, assuming normal (0)")
        return 0  # Default to normal

def extract_wavelet_features(signal, target_length=None):
    """
    Extract wavelet features from the signal and ensure consistent length.
    """
    # Apply 1-level Discrete Wavelet Transform using Daubechies 4 wavelet
    (cA, cD) = pywt.dwt(signal, 'db4')
    
    # Combine approximation and detail coefficients
    features = np.concatenate([cA, cD])
    
    # Ensure consistent feature length
    if target_length is not None:
        if len(features) > target_length:
            # Truncate if too long
            features = features[:target_length]
        elif len(features) < target_length:
            # Pad with zeros if too short
            features = np.pad(features, (0, target_length - len(features)), mode='constant')
    
    return features

def preprocess_and_save_wavelet_features(data_dir, save_dir, set_name):
    """
    Processes all .wav files in a directory, applies DWT, and saves the coefficients with labels.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Scanning directory: {data_dir}")
    
    # Load reference labels for this training set
    reference_labels = load_reference_labels(data_dir)
    print(f"  Loaded {len(reference_labels)} reference labels")
    
    processed_count = 0
    normal_count = 0
    abnormal_count = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            file_id = os.path.splitext(filename)[0]
            
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=2000)
                
                # Extract wavelet features with consistent length
                features = extract_wavelet_features(y, TARGET_FEATURE_LENGTH)
                
                # Get label for this file from reference
                label = get_label_from_filename_and_set(filename, set_name, reference_labels)
                
                # Count labels
                if label == 0:
                    normal_count += 1
                else:
                    abnormal_count += 1
                
                # Save features and label to a .npz file
                save_path = os.path.join(save_dir, f"{file_id}.npz")
                np.savez(save_path, 
                        features=features,  # Consistent feature vector
                        label=label,        # Classification label (0: normal, 1: abnormal)
                        original_cA_shape=len(pywt.dwt(y, 'db4')[0]),  # For reference
                        original_cD_shape=len(pywt.dwt(y, 'db4')[1]),  # For reference
                        set_name=set_name,  # Which training set this came from
                        filename=filename)  # Original filename
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} files...")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"  Total processed: {processed_count} files from {set_name}")
    print(f"  Normal: {normal_count}, Abnormal: {abnormal_count}")

if __name__ == '__main__':
    print("Starting wavelet preprocessing...")
    print(f"Target feature length: {TARGET_FEATURE_LENGTH}")
    
    # Ensure the main save directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    total_processed = 0
    
    # Process each training set
    for set_name in TRAINING_SETS:
        data_directory = os.path.join(BASE_PROJECT_DIR, set_name)
        if os.path.exists(data_directory):
            print(f"\n--- Processing training set: {set_name} ---")
            preprocess_and_save_wavelet_features(data_directory, SAVE_DIR, set_name)
        else:
            print(f"Directory not found, skipping: {data_directory}")

    # Count total files processed and label distribution
    if os.path.exists(SAVE_DIR):
        npz_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.npz')]
        total_processed = len(npz_files)
        
        # Count label distribution
        total_normal = 0
        total_abnormal = 0
        
        for npz_file in npz_files[:100]:  # Sample first 100 files for quick count
            try:
                sample_path = os.path.join(SAVE_DIR, npz_file)
                sample_data = np.load(sample_path)
                if sample_data['label'] == 0:
                    total_normal += 1
                else:
                    total_abnormal += 1
                sample_data.close()
            except:
                pass

    print(f"\nWavelet preprocessing complete.")
    print(f"Total files processed: {total_processed}")
    print(f"Sample label distribution (first 100): Normal: {total_normal}, Abnormal: {total_abnormal}")
    print(f"Processed data saved in: {SAVE_DIR}")
    
    # Quick verification
    if total_processed > 0:
        print(f"\nQuick verification of the first file:")
        first_file = npz_files[0]
        sample_path = os.path.join(SAVE_DIR, first_file)
        sample_data = np.load(sample_path)
        print(f"Keys: {list(sample_data.keys())}")
        print(f"Features shape: {sample_data['features'].shape}")
        print(f"Label: {sample_data['label']} ({'Normal' if sample_data['label'] == 0 else 'Abnormal'})")
        print(f"Set name: {sample_data['set_name']}")
        sample_data.close()
