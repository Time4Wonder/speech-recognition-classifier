import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import sklearn

import code_extract_mfccs
import code_train

#import tensorflow as tf
#import tensorflow_datasets as tfds

# classic read of dataset
audiopath= "data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"
audio_data, sample_rate = sf.read(audiopath)

# print the audio data and sample rate
print("Sample rate:" , sample_rate)
print("Number of samples:" , len(audio_data))
print("Duration (seconds):" , len(audio_data) / sample_rate)

#  Visualize the audio data
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y=audio_data, sr=sample_rate)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()


# Example of MFCC extraction and visualization
mfccs = code_extract_mfccs.extract_mfccs(audiopath)
print(f"Shape of the MFCCs: {mfccs.shape}")
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate, hop_length=512)
plt.colorbar()
plt.title('MFCCs')
plt.tight_layout()
plt.show()

# Prepare the dataset using the function from code_train.py
code_train.prepare_dataset_scikit_learn('data/speech_commands_v0.02')
