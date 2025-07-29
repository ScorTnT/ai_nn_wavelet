import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Note: This script requires the following libraries to be installed:
# pip install numpy pandas scikit-learn librosa

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=2000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def train_model():
    """Trains a model to classify heart sounds."""
    # Load labels
    labels_df = pd.read_csv('../training-a/REFERENCE.csv', header=None, names=['filename', 'label'])
    labels_dict = dict(zip(labels_df.filename, labels_df.label))

    # Prepare data
    features = []
    labels = []
    data_dir = '../training-a'
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_id = os.path.splitext(filename)[0]
            if file_id in labels_dict:
                audio_path = os.path.join(data_dir, filename)
                mfccs = extract_features(audio_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(labels_dict[file_id])

    X = np.array(features)
    y = np.array(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='auto')
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    train_model()
