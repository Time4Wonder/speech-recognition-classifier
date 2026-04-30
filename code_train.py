import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import code_extract_mfccs as cemfccs

# Main directory of the dataset 
data_dir = 'data/speech_commands_v0.02'

def prepare_dataset_scikit_learn(data_directory, sr=16000, n_mfcc=13):
    """
    Prepare dataset for scikit-learn with progress bar during audio file loading.
    
    In this example ten commands are used for the training.
    As a first step we recommend to start with a binary classification. For this remove the
    loop or reduce the list of allowed words.
    """
    
    allowed_words = ['yes', 'no']
    X = []
    y = []
    
    # First pass: collect all matching audio files
    audio_files = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.wav'):
                label = os.path.basename(root)
                if label in allowed_words:
                    file_path = os.path.join(root, file)
                    audio_files.append((file_path, label))
    
    # Load audio files with progress bar
    with tqdm(total=len(audio_files), desc="Processing audio files", unit="file") as pbar:
        for file_path, label in audio_files:
            try:
                mfccs = cemfccs.extract_mfccs(file_path, sr=sr, n_mfcc=n_mfcc)
                mfccs_mean = np.mean(mfccs, axis=1)
                X.append(mfccs_mean)
                y.append(label)
            except Exception as e:
                tqdm.write(f"Error processing {file_path}: {e}")
            finally:
                pbar.update(1)

    # Convert to numpy arrays after collecting all files
    X = np.array(X)
    y = np.array(y)

    if X.size == 0:
        raise ValueError(f"No audio samples found in {data_directory}")

    # Convert labels to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # If we only have a single sample, return it as the training set and empty others
    if len(X) == 1:
        X_train = X
        y_train = y_encoded
        n_features = X.shape[1] if X.ndim == 2 else 1
        X_val = np.empty((0, n_features))
        X_test = np.empty((0, n_features))
        y_val = np.empty((0,), dtype=y_encoded.dtype)
        y_test = np.empty((0,), dtype=y_encoded.dtype)
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

    # First split: train vs (val+test)
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
    except ValueError:
        # Fallback to non-stratified split when stratification is not possible
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=None
        )

    # If there is 0 or 1 sample left in X_temp, assign appropriately
    if len(X_temp) == 0:
        print("Length of X_temp == 0. Assign appropriately")
        n_features = X.shape[1] if X.ndim == 2 else 1
        X_val = np.empty((0, n_features))
        X_test = np.empty((0, n_features))
        y_val = np.empty((0,), dtype=y_encoded.dtype)
        y_test = np.empty((0,), dtype=y_encoded.dtype)
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

    if len(X_temp) == 1:
        # Put the single remaining sample into validation and leave test empty
        print("Length of X_temp == 1.")
        X_val = X_temp
        y_val = y_temp
        n_features = X.shape[1] if X.ndim == 2 else 1
        X_test = np.empty((0, n_features))
        y_test = np.empty((0,), dtype=y_encoded.dtype)
        return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

    # Split the temporary set into validation and test
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
        )

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

# execution and printing results
print("Prepare data for scikit-learn...")
# Hinweis: Du musst sicherstellen, dass die Funktion extract_mfccs definiert ist, 
# bevor du prepare_dataset_scikit_learn aufrufst.
X_train_sk, X_val_sk, X_test_sk, y_train_sk, y_val_sk, y_test_sk, label_encoder_sk = \
    prepare_dataset_scikit_learn(data_dir)

print(f"Training data Shape (scikit-learn): {X_train_sk.shape}, Labels: {y_train_sk.shape}")
print(f"Validation data Shape (scikit-learn): {X_val_sk.shape}, Labels: {y_val_sk.shape}")
print(f"Test Data Shape (scikit-learn): {X_test_sk.shape}, Labels: {y_test_sk.shape}")
print(f"Classes: {label_encoder_sk.classes_}")