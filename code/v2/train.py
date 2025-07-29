
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Note: This script requires the following libraries to be installed:
# pip install numpy pandas scikit-learn librosa

def extract_features(audio_path):
    """Extracts a variety of features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=2000)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent.T, axis=0)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr.T, axis=0)

        # Concatenate all features
        features = np.concatenate((mfccs_mean, chroma_mean, cent_mean, zcr_mean))
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def train_model():
    """Trains a model to classify heart sounds using more features and a RandomForestClassifier."""
    all_features = []
    all_labels = []
    training_sets = ['a', 'b', 'c', 'd', 'e', 'f']

    for set_id in training_sets:
        data_dir = f'/workspace/training-{set_id}'
        #/workspace/training-a
        labels_path = os.path.join(data_dir, 'REFERENCE.csv')
        print(labels_path)
        if not os.path.exists(labels_path):
            print(f"Labels file not found for training set {set_id}, skipping.")
            continue

        labels_df = pd.read_csv(labels_path, header=None, names=['filename', 'label'])
        labels_dict = dict(zip(labels_df.filename, labels_df.label))

        print(f"Processing training set {set_id}...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.wav'):
                file_id = os.path.splitext(filename)[0]
                if file_id in labels_dict:
                    audio_path = os.path.join(data_dir, filename)
                    features = extract_features(audio_path)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(labels_dict[file_id])

    X = np.array(all_features)
    y = np.array(all_labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    train_model()
