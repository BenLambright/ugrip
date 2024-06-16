import numpy as np
import pandas as pd

import textgrids

FRAME_DURATION = 30 # 30 msec
OVERLAP_RATE = 0 # frames don't overlap

def readFile(path):
    '''
    Read the file and return the list of SPEECH/NONSPEECH labels for each frame
    '''
        
    labeled_list  = []
    grid = textgrids.TextGrid(path)

    for interval in grid['silences']:
        if interval.text == "-" or interval.text == " ":
            label = 0
        else:
            label = 1
        dur = interval.dur
        dur_msec = dur * 1000 # sec -> msec
        num_frames = int(round(dur_msec /30)) # the audio is divided into 30 msec frames
        print(dur_msec)
        for i in range(num_frames):
            
            labeled_list.append(label)

    return labeled_list


# Function for reading labels from .TextGrig file:
def readLabels(path, sample_rate):
        
    labeled_list  = []
    grid = textgrids.TextGrid(path)

    for interval in grid['silences']:
        if interval.text == "-" or interval.text == " ":
            label = 0
        else:
            label = 1

        dur = interval.dur
        dur_samples = int(np.round(dur * sample_rate)) # sec -> num of samples
        
        for i in range(dur_samples):
            labeled_list.append(label)

    return labeled_list

import os
import librosa

# root = 'Female/TMIT/SA2'
# audio_name = 'FF482278-24C9-4FB0-A7BE-FD92D1FE17B4-1430028319-1.0-m-26-hu.wav'
# annotation_name = 'FF482278-24C9-4FB0-A7BE-FD92D1FE17B4-1430028319-1_0-m-26-hu.TextGrid'
# audio_path = os.path.join('/home/benjamin.lambright/Desktop/ugrip/Datasets_to_sort/hungry', audio_name)
# annotation_path = os.path.join('/home/benjamin.lambright/Desktop/ugrip/Datasets_to_sort/hungry', annotation_name)

annotation_path = "/Users/mo-alowais/Documents/master_annotations/0a983cd2-0078-4698-a048-99ac01eb167a-1433917038889-1.7-f-04-hu.wav.TextGrid"
audio_path = "/Users/mo-alowais/Documents/baby_audio/0a983cd2-0078-4698-a048-99ac01eb167a-1433917038889-1.7-f-04-hu.wav"

# Read annotation
label_list = readFile(annotation_path)
# Read wav file
data, fs = librosa.load(audio_path)

# define time axis
Ns = len(data)  # number of sample
Ts = 1 / fs  # sampling period
t = np.arange(Ns) * 1000 * Ts  # time axis


shift = 1 - OVERLAP_RATE
frame_length = int(np.floor(FRAME_DURATION * fs / 1000)) # frame length in sample
frame_shift = round(frame_length * shift) # frame shift in sample


# Plot:

# import matplotlib.pyplot as plt

# figure = plt.Figure(figsize=(10, 7), dpi=85)
# plt.plot(t, data)

# for i, frame_labeled in enumerate(label_list):
#     idx = i * frame_shift
#     if (frame_labeled == 1):
#         plt.axvspan(xmin= t[idx], xmax=t[idx + frame_length-1], ymin=-1000, ymax=1000, alpha=0.4, zorder=-100, facecolor='g', label='Speech')

# plt.title("Ground truth labels")
# plt.legend(['Signal', 'Speech'])
# plt.show()

# Preparation
import python_speech_features
from tqdm import tqdm 
from sklearn import model_selection, preprocessing, metrics
# from keras.models import Model
# import keras.layers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Function for getting all files in directories and sub-directories woth definite extension
def getFiles(path, extension):
    list_paths = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if(file.endswith(extension)):
                list_paths.append(os.path.join(root, file))
    return list_paths

annotation_path = '/Users/mo-alowais/Documents/master_annotations'
annotation_extension = '.TextGrid'
audio_path = '/Users/mo-alowais/Documents/baby_audio'
audio_extension = '.wav'

annotation_files = getFiles(path=annotation_path, extension=annotation_extension)
audio_files = getFiles(path=audio_path, extension=audio_extension)
annotation_files = sorted(annotation_files)
audio_files = sorted(audio_files)
# Processing files

# Set params for Mel-Frequency Cepstral Coefficient (MFCC)
# preemphasis_coef (Pre-emphasis coefficient): The coefficient is used in a preemphasis filter applied to the audio signal before further processing. A value close o 1 emphasis higher frequencies, compensating for the natural roll-off of high frequencies in audio signals
# frame_length: The length of time each window will be when the signal is segmented into frames. (Normally ranged between 20 to 30 milliseconds)
# fram_step: The time window shifted between frames.
# window_function: Applied to each frame to reduce spectral leakage from outside the frame. Hamming window is commonly used
# num_nfft (Number of points for FFT): The number of points used in the Fast Fourier Transform when converting each frame from the time domain to the frequency domain. Large FFT size allow for more precise frequency resolution but also increased computation time.
# num_features (Number of Mel filters): The number of Mel-frequency bins used in the MFCC.

# Mel scale is a way of representing frequencies in a manner similar to human perception. By using Mel filters, the features capture the psychoacoustic properties of the sound.

preemphasis_coef = 0.97 
frame_length = 0.025
frame_step = 0.01
window_function = np.hamming
num_nfft = 551
# num_nfft = 551 
num_features = 32 

import librosa

# Extraction features for each file:
for i in tqdm(range(len(audio_files))):
    sig, sample_rate = librosa.load(audio_files[i])
    markers = readLabels(path=annotation_files[i], sample_rate=sample_rate)

    # Extract features:
    # features_fbank (Mel-frequency cepstral coefficiens): captures the spectral information of the audio signal in a way that mimics human auditory perception
    # feature_energy: the overall energy of the audio signal within a specific frequency range
    features_fbank, feature_energy  = python_speech_features.base.fbank(signal=sig, 
                                                                        samplerate=sample_rate, 
                                                                        winlen=frame_length, 
                                                                        winstep=frame_step, 
                                                                        nfilt=num_features, 
                                                                        nfft=num_nfft, 
                                                                        lowfreq=0, 
                                                                        highfreq=None, 
                                                                        preemph=preemphasis_coef, 
                                                                        winfunc=window_function)
    # Logfbank and log energy:
    features_logfbank = np.log(features_fbank)
    feature_logenergy = np.log(feature_energy)
    # print('Shape logfbank:', features_fbank.shape)
    # print('Shape logenergy:', feature_logenergy.shape)

    # Merge logfbank and log energy:
    features = np.hstack((feature_logenergy.reshape(feature_logenergy.shape[0], 1), features_logfbank))
    # print('Shape features:', features.shape)

    # Reshape labels for each group of features:
    markers_of_frames = python_speech_features.sigproc.framesig(sig=markers, 
                                                                frame_len=frame_length * sample_rate, 
                                                                frame_step=frame_step * sample_rate, 
                                                                winfunc=np.ones)
    # print('Shape markers_of_frame:', markers_of_frames.shape)
    # For every frame calc label:
    # For every frame calc label:
    marker_per_frame = np.zeros(markers_of_frames.shape[0])
    marker_per_frame = np.array([1 if np.sum(markers_of_frames[j], axis=0) > markers_of_frames.shape[0] / 2 else 0 for j in range(markers_of_frames.shape[0])])
    marker_per_frame = np.zeros(markers_of_frames.shape[0])
    marker_per_frame = np.array([1 if np.sum(markers_of_frames[j], axis=0) > markers_of_frames.shape[0] / 2 else 0 for j in range(markers_of_frames.shape[0])])

    # Create massive for stacking features in first step:
    if i == 0:
        dataset_tmp = np.zeros((1, num_features + 2))

    # Check indices of features and labels:
    restrictive_index = np.min([features.shape[0], marker_per_frame.shape[0]], axis=0)
    features_tmp = features[:restrictive_index]
    marker_per_frame_tmp = marker_per_frame[:restrictive_index]
    
    # Merge label and franes and all frames in dataset:
    dataset_tmp = np.vstack((dataset_tmp, np.hstack((marker_per_frame_tmp.reshape(marker_per_frame_tmp.shape[0], 1), features_tmp))))

    dataset = dataset_tmp[1:]
    # Preparation of data
# Split dataset on train and test:
X = dataset[:, 1:]
y = dataset[:, 0]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, shuffle=True, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scale data:
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape datasets to 1 x 300 x num_features:
# So, each group need to consist of 300 frames:
n_frames = 300
X_train_reshaped = X_train[:int(X_train.shape[0] / n_frames) * n_frames]
X_train_reshaped = X_train_reshaped.reshape(int(X_train_reshaped.shape[0] / n_frames), n_frames, X_train_reshaped.shape[1])
X_test_reshaped = X_test[:int(X_test.shape[0] / n_frames) * n_frames]
X_test_reshaped = X_test_reshaped.reshape(int(X_test_reshaped.shape[0] / n_frames), n_frames, X_test_reshaped.shape[1])

# Encoding label:
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train_reshaped = y_train[:int(y_train.shape[0] / n_frames) * n_frames]
y_train_reshaped = y_train_reshaped.reshape(int(y_train_reshaped.shape[0] / n_frames), n_frames, y_train_reshaped.shape[1])
y_test_reshaped = y_test[:int(y_test.shape[0] / n_frames) * n_frames]
y_test_reshaped = y_test_reshaped.reshape(int(y_test_reshaped.shape[0] / n_frames), n_frames, y_test_reshaped.shape[1])

print(y_train_reshaped.shape)
X_train_reshaped = X_train_reshaped.reshape((703, 300, 33, 1))

def temporal_contrastive_loss(model, predictions, targets, idx, alpha1=0.1, alpha2=0.03):
    print("pred shape: ",predictions.shape)
    print("target shape: ", targets.shape)

    bce_loss = F.binary_cross_entropy(predictions, targets)
    
    coherence_loss = 0
    for j in range(1, targets.shape[0]):
        for i in range(1, targets.shape[1]):
            delta_target = (targets[j, i] - targets[j, i - 1]).abs()
            delta_rep = (model.intermediate_rep[j][i] - model.intermediate_rep[j][i - 1]).pow(2).sum(dim=1)
        
            coherence_loss += alpha1 * (delta_target > 0).float() * delta_rep
            coherence_loss -= alpha2 * (delta_target == 0).float() * delta_rep
    
    coherence_loss = coherence_loss.mean()
    total_loss = bce_loss + coherence_loss
    return total_loss




class CRNN_modified(nn.Module):
    def __init__(self, time_steps=703, freq_bins=300, input_channels=33,num_classes=2):
        super(CRNN_modified, self).__init__()
        self.time_steps = time_steps
        self.freq_bins = freq_bins
        self.channels = input_channels
        self.intermediate_rep = [[[0]]]

        self.cnn = nn.Sequential(
            nn.Conv2d(300, 16, kernel_size=3, stride=1, padding=1),
            nn.GLU(dim=1),
            # nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),  # change 16 to 8 due to GLU halving channels
            nn.GLU(dim=1),
            # nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  # change 32 to 16 due to GLU halving channels
            nn.GLU(dim=1),
            # nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),  # change 64 to 32 due to GLU halving channels
            nn.GLU(dim=1),
            # nn.MaxPool2d((1, 4)),
            
            nn.Conv2d(64, 600, kernel_size=3, stride=1, padding=1),  # change 128 to 64 due to GLU halving channels
            nn.GLU(dim=1),
            # nn.MaxPool2d((1, 4))
        )
        
        
        self.rnn = nn.GRU(33, 33, num_layers=1, bidirectional=False, batch_first=False)
        self.fc = nn.Linear(33, num_classes)
    
    def forward(self, x):
        
        # x = self.cnn(x)
        
        # x = x.permute(0, 2, 1, 3)  # change to (batch_size, time, channels, freq_bins)
       
        # x = x.reshape(batch_size, time_steps, channels * freq_bins)
        # x, _ = self.rnn(x)
        
        # x = self.fc(x)
        batch_size = x.size(0)
        x = self.cnn(x)
        
                
        print("shape: ",x.shape)
        x = x.reshape(703, 300, 33)
        self.intermediate_rep.append(x)
        x, _ = self.rnn(x)
        x = self.fc(x)


        x = torch.sigmoid(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.tensor(X_train_reshaped, dtype=torch.float).to(device).shape

from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Define input and output dimensions
input_dim = 33
output_dim = 2

# Initialize model


# model = TransformerModel(input_dim=input_dim, output_dim=output_dim).to(device)
model = CRNN_modified().to(device)
# Define input and output data
target_tensor = torch.tensor(y_train_reshaped, dtype=torch.float).to(device)
query_input_tensor = torch.tensor(X_train_reshaped, dtype=torch.float).to(device)

# input_data = torch.randn(587, 300, input_dim)

# loss function and optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# define cosine decay
T_0 = 10  # Initial number of epochs for the first restart
T_mult = 2  # Multiplicative factor by which the number of epochs for every subsequent restarts increases
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

# epoch num
num_epochs = 10

index_loss = 1 # for intermediate function
for epoch in range(num_epochs):
    # Forward pass
    output = model(query_input_tensor)

    print(output.shape)
    print("target tensor: ",target_tensor.shape)
    # Compute the loss

    loss = temporal_contrastive_loss(model, output, target_tensor, index_loss)
    index_loss += 1
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update the learning rate
    scheduler.step()

    # Print loss for monitoring
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Optionally, calculate and print accuracy
    with torch.no_grad():
        predicted_labels = torch.argmax(output, dim=-1)
        target_labels = torch.argmax(target_tensor, dim=-1)
        accuracy = accuracy_score(target_labels.view(-1).cpu().numpy(), predicted_labels.view(-1).cpu().numpy())
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.4f}')