import yaml
import whisperx
import librosa
from scipy.signal import fftconvolve
import torch
import h5py
import numpy as np
import unicodedata
import re

SAMPLE_RATE = 16000

# Load config file
def read_config_file(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Add echo
def add_echo(aud_fname, rir_fname):
    audio, sr = librosa.load(aud_fname, sr=SAMPLE_RATE)
    rir, sr = librosa.load(rir_fname, sr=SAMPLE_RATE)

    augmented = fftconvolve(audio, rir)
    return augmented

# Extract whisper embeddings
def encode_whisper_embeddings(aud, model):
    audio = {}
    audio['inputs'] = aud
    feats = model.preprocess(audio)['inputs']

    embeddings = model.model.encode(feats)
    return torch.as_tensor(embeddings)

# Load embeddings from hdf5 file
def load_embeddings_and_rir_from_hdf5(hdf5_file_path, audio_file):
    with h5py.File(hdf5_file_path, 'r') as hf:
        # Access the group corresponding to the audio file
        if audio_file in hf:
            group = hf[audio_file]
            embedding_original = np.array(group['embedding_original'])
            embedding_with_echo = np.array(group['embedding_with_echo'])

            return embedding_original, embedding_with_echo
        else:
            print(f"{audio_file} not found in HDF5 file.")
            return None, None
        
# Normalise string
def normalize_text(s):
    # Convert to lower case
    s = s.lower()
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    
    # Additional replacements
    s = s.replace('ß', 'ss')
    s = s.replace('ö', 'oe')
    s = s.replace('ä', 'ae')
    s = s.replace('ü', 'ue')
    
    # Convert umlauts and special characters to normal English characters
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8')
    
    return s

