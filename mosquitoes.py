# %% [code]
# WINGBEATS DATABASE

from __future__ import division
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import lightgbm as lgb
import xgboost
import librosa
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import zipfile
import soundfile as sf
from scipy import signal
import seaborn as sn
import os

def get_data(zip_path, target_names):
    X = []                    # holds all data
    y = []                    # holds all class labels
    filenames = []            # holds all the file names
    target_count = []         # holds the counts in a class

    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for i, target in enumerate(target_names):
            target_count.append(0)  # initialize target count
            target_path = 'Wingbeats/' + target + '/'  # path inside zip

            # Get all .wav files for the target species
            for file in zip_ref.namelist():
                if file.startswith(target_path) and file.endswith('.wav'):
                    with zip_ref.open(file) as wav_file:
                        data, fs = sf.read(wav_file)
                        X.append(data)
                        y.append(i)
                        filenames.append(file)
                        target_count[i] += 1
                        if target_count[i] > 20000:
                            break
            print(target, '#recs = ', target_count[i])

    X = np.vstack(X)
    y = np.hstack(y)
    X = X.astype("float32")

    print("\nTotal dataset size:")
    print('# of classes:', len(np.unique(y)))
    print('total dataset size:', X.shape[0])
    print('Sampling frequency =', fs)
    print("n_samples:", X.shape[1])
    print("duration (sec):", X.shape[1] / fs)
    return X, y

# main
zip_path = 'C:\\Users\\shani\\DRDO\\archive (2).zip'  # replace with your actual zip file path
fs = 8000
target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X, y = get_data(zip_path, target_names)
X, y = shuffle(X, y, random_state=2018)

names = ["XGBoost", "Random Forest", "ExtraTreesClassifier", "Linear SVM", "RBF SVM"]
classifiers = [
    xgboost.XGBClassifier(n_estimators=650, learning_rate=0.2),
    RandomForestClassifier(n_estimators=650, min_samples_split=3, min_samples_leaf=2, random_state=2018, n_jobs=-1),
    ExtraTreesClassifier(n_estimators=650, random_state=2018, n_jobs=-1),
    SVC(kernel="linear", C=0.01),
    SVC(gamma=0.008, C=0.1)
]

# Transform the data
XX = np.zeros((X.shape[0], 129), dtype="float32")  # allocate space
for i in range(X.shape[0]):
    XX[i] = 10 * np.log10(signal.welch(X[i], fs=fs, window='hann', nperseg=256, noverlap=128+64)[1])

# Show one recording
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(np.linspace(0, X.shape[1] / fs, X.shape[1]), X[0])
plt.xlabel('time (s)')
plt.title('Wingbeat recording')

plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, fs / 2, 129), XX[0])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Welch Power Spectral Density')
plt.show()

# A quick result
X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.20, random_state=2018)
model = xgboost.XGBClassifier(n_estimators=650, learning_rate=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
ac = accuracy_score(y_test, y_pred)
print("Name: XGBoost, Accuracy:", ac)

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
plt.figure(figsize=(15, 10))
sn.heatmap(df_cm, annot=True)
plt.show()
