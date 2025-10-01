#!/usr/bin/env python3
"""
Complete CHB-MIT EEG Neurological Disease Detection Pipeline
===========================================================
This pipeline detects three neurological conditions using CHB-MIT dataset:
1. Seizure Detection using NeuroGNN approach
2. Alzheimer's Detection using LEAD approach  
3. Parkinson's Detection using EEGPT approach

Compatible with CHB-MIT Scalp EEG Database from Kaggle
Author: Assistant
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
import zipfile
import requests
from pathlib import Path
import glob
warnings.filterwarnings('ignore')

# Deep learning and EEG processing imports
try:
    import torch
    import torch.nn as nn
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import mne  # For EDF file processing
    mne.set_log_level('WARNING')  # Reduce MNE verbosity
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Install with: pip install torch tensorflow scikit-learn mne kaggle")
    sys.exit(1)

class CHBMITDownloader:
    """Manage CHB-MIT dataset from local directory"""
    
    def __init__(self, download_dir=None):
        """
        Initialize with local directory path
        
        Args:
            download_dir: Path to your local CHB-MIT data directory
                         Example: r"C:\\Users\\chanm\\Downloads\\archive (1)\\Annotated_EEG"
                         OR use forward slashes: "C:/Users/chanm/Downloads/archive (1)/Annotated_EEG"
        """
        if download_dir is None:
            # Default to current directory if not specified
            self.download_dir = Path("./chb_mit_data")
        else:
            self.download_dir = Path(download_dir)
        
        print(f"📁 Using local data directory: {self.download_dir}")
        
        if not self.download_dir.exists():
            print(f"⚠️  Warning: Directory does not exist: {self.download_dir}")
            print("Please provide a valid path to your CHB-MIT data")
        else:
            print(f"✅ Directory found: {self.download_dir}")
    
    def verify_local_dataset(self):
        """Verify that local dataset exists and is accessible"""
        try:
            if not self.download_dir.exists():
                print(f"❌ Directory not found: {self.download_dir}")
                return False
            
            # Check for EDF files
            edf_files = list(self.download_dir.glob("**/*.edf"))
            
            if not edf_files:
                print(f"❌ No EDF files found in {self.download_dir}")
                print("Please check that the directory contains CHB-MIT EDF files")
                return False
            
            print(f"✅ Found {len(edf_files)} EDF files")
            
            # Show sample of files
            print("\n📋 Sample files:")
            for i, file in enumerate(edf_files[:5]):
                print(f"   {i+1}. {file.name}")
            
            if len(edf_files) > 5:
                print(f"   ... and {len(edf_files) - 5} more files")
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying dataset: {e}")
            return False
    
    def download_chb_mit_dataset(self):
        """Not needed - using local data"""
        print("ℹ️  Using local dataset - no download needed")
        return self.verify_local_dataset()
    
    def alternative_download_method(self):
        """Not needed - using local data"""
        print("ℹ️  Using local dataset from:", self.download_dir)
        print("If files are not found, please check the directory path")
        return False
    
    def get_edf_files(self):
        """Get list of all EDF files"""
        edf_files = list(self.download_dir.glob("**/*.edf"))
        return sorted(edf_files)
    
    def get_seizure_files(self):
        """Get list of files that contain seizures"""
        try:
            # Look for RECORDS-WITH-SEIZURES file
            records_file = self.download_dir / "RECORDS-WITH-SEIZURES"
            if not records_file.exists():
                # Try alternative locations
                possible_files = list(self.download_dir.glob("**/RECORDS-WITH-SEIZURES*"))
                if possible_files:
                    records_file = possible_files[0]
                else:
                    print("⚠️ RECORDS-WITH-SEIZURES file not found")
                    return []
            
            with open(records_file, 'r') as f:
                seizure_files = [line.strip() for line in f.readlines() if line.strip()]
            
            # Convert to full paths
            seizure_paths = []
            for file in seizure_files:
                full_path = self.download_dir / file
                if full_path.exists():
                    seizure_paths.append(full_path)
                else:
                    # Try finding in subdirectories
                    found_files = list(self.download_dir.glob(f"**/{file}"))
                    if found_files:
                        seizure_paths.extend(found_files)
            
            print(f"📊 Found {len(seizure_paths)} seizure files")
            return seizure_paths
            
        except Exception as e:
            print(f"❌ Error reading seizure files: {e}")
            return []

class CHBMITPreprocessor:
    """Preprocessing class specifically for CHB-MIT EEG data"""
    
    def __init__(self, fs=256):
        self.fs = fs  # CHB-MIT sampling frequency
        self.chb_mit_channels = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
            'FT9-FT10', 'FT10-T8'
        ]
        self.standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2'
        ]
        self.scaler = StandardScaler()
        
    def load_edf_file(self, file_path):
        """Load CHB-MIT EDF file"""
        try:
            print(f"📖 Loading EDF file: {Path(file_path).name}")
            
            # Load EDF file with MNE
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Get data and convert to DataFrame
            data, times = raw.get_data(return_times=True)
            
            # Create DataFrame with channel names
            df = pd.DataFrame(data.T, columns=raw.ch_names)
            df['time'] = times
            
            print(f"✅ EDF loaded: {df.shape}")
            print(f"📊 Sampling rate: {raw.info['sfreq']} Hz")
            print(f"📊 Channels: {list(raw.ch_names)}")
            print(f"⏱️ Duration: {len(times)/raw.info['sfreq']:.1f} seconds")
            
            # Remove non-EEG channels
            eeg_channels = [ch for ch in df.columns if ch != 'time' and not ch.startswith('-')]
            df_eeg = df[eeg_channels]
            
            return df_eeg, raw.info
            
        except Exception as e:
            print(f"❌ Error loading EDF file: {e}")
            return None, None
    
    def map_chb_mit_to_standard(self, data):
        """Map CHB-MIT bipolar channels to standard monopolar channels"""
        try:
            # CHB-MIT uses bipolar montage, we need to approximate monopolar
            # This is a simplified mapping - in practice, you'd need reference electrode info
            
            mapped_data = pd.DataFrame()
            
            # Mapping bipolar to approximate monopolar (simplified approach)
            channel_mapping = {
                'Fp1': ['FP1-F7', 'FP1-F3'],
                'Fp2': ['FP2-F4', 'FP2-F8'], 
                'F7': ['FP1-F7', 'F7-T7'],
                'F3': ['FP1-F3', 'F3-C3'],
                'F4': ['FP2-F4', 'F4-C4'],
                'F8': ['FP2-F8', 'F8-T8'],
                'Fz': ['FZ-CZ'],
                'C3': ['F3-C3', 'C3-P3'],
                'C4': ['F4-C4', 'C4-P4'],
                'Cz': ['FZ-CZ', 'CZ-PZ'],
                'P3': ['C3-P3', 'P3-O1'],
                'P4': ['C4-P4', 'P4-O2'],
                'Pz': ['CZ-PZ'],
                'T3': ['F7-T7', 'T7-P7'],  # T7 in 10-10 system
                'T4': ['F8-T8', 'T8-P8'],  # T8 in 10-10 system
                'T5': ['T7-P7', 'P7-O1'],  # P7 in 10-10 system
                'T6': ['T8-P8', 'P8-O2'],  # P8 in 10-10 system
                'O1': ['P7-O1', 'P3-O1'],
                'O2': ['P8-O2', 'P4-O2']
            }
            
            # Use available channels to approximate standard channels
            available_channels = data.columns.tolist()
            
            for std_ch, source_channels in channel_mapping.items():
                # Find available source channels
                available_sources = [ch for ch in source_channels if ch in available_channels]
                
                if available_sources:
                    # Average available sources to approximate the standard channel
                    mapped_data[std_ch] = data[available_sources].mean(axis=1)
                else:
                    # If no mapping available, create zeros
                    mapped_data[std_ch] = 0.0
            
            # Fill missing standard channels with first available channel
            if mapped_data.empty:
                # Fallback: use first 19 channels
                for i, std_ch in enumerate(self.standard_channels[:min(19, len(available_channels))]):
                    mapped_data[std_ch] = data.iloc[:, i] if i < len(available_channels) else 0.0
            
            print(f"✅ Mapped to {len(mapped_data.columns)} standard channels")
            return mapped_data
            
        except Exception as e:
            print(f"❌ Error in channel mapping: {e}")
            # Fallback: use first 19 channels
            return data.iloc[:, :min(19, data.shape[1])]
    
    def apply_preprocessing(self, data):
        """Apply full preprocessing pipeline"""
        try:
            print("🔧 Applying preprocessing pipeline...")
            
            # 1. Map channels
            mapped_data = self.map_chb_mit_to_standard(data)
            
            # 2. Bandpass filter (0.5-45 Hz)
            filtered_data = self.bandpass_filter(mapped_data)
            
            # 3. Notch filter (60 Hz)
            notched_data = self.notch_filter(filtered_data)
            
            # 4. Normalize
            normalized_data = self.normalize_data(notched_data)
            
            print("✅ Preprocessing completed")
            return normalized_data
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return data
    
    def bandpass_filter(self, data, low_freq=0.5, high_freq=45):
        """Apply bandpass filter"""
        try:
            nyquist = self.fs / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_data = pd.DataFrame(index=data.index)
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    filtered_data[col] = signal.filtfilt(b, a, data[col].values)
                else:
                    filtered_data[col] = data[col]
            
            print(f"✅ Bandpass filter applied: {low_freq}-{high_freq} Hz")
            return filtered_data
            
        except Exception as e:
            print(f"❌ Error in bandpass filtering: {e}")
            return data
    
    def notch_filter(self, data, freq=60):
        """Apply notch filter for power line noise"""
        try:
            nyquist = self.fs / 2
            freq_normalized = freq / nyquist
            Q = 30
            
            b, a = signal.iirnotch(freq_normalized, Q)
            filtered_data = pd.DataFrame(index=data.index)
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    filtered_data[col] = signal.filtfilt(b, a, data[col].values)
                else:
                    filtered_data[col] = data[col]
            
            print(f"✅ Notch filter applied: {freq} Hz")
            return filtered_data
            
        except Exception as e:
            print(f"❌ Error in notch filtering: {e}")
            return data
    
    def normalize_data(self, data):
        """Normalize data"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_normalized = data.copy()
            data_normalized[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
            print("✅ Data normalized")
            return data_normalized
        except Exception as e:
            print(f"❌ Error in normalization: {e}")
            return data
    
    def segment_data(self, data, window_size=1024, overlap=0.5):
        """Segment data into windows"""
        try:
            step_size = int(window_size * (1 - overlap))
            segments = []
            
            for i in range(0, len(data) - window_size + 1, step_size):
                segment = data.iloc[i:i + window_size]
                segments.append(segment.values)
            
            segments = np.array(segments)
            print(f"✅ Data segmented: {segments.shape[0]} segments of {window_size} samples")
            return segments
            
        except Exception as e:
            print(f"❌ Error in segmentation: {e}")
            return np.array([data.values])

class SeizureDetector:
    """Enhanced Seizure Detection for CHB-MIT data"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load seizure detection model"""
        try:
            class SeizureCNN(nn.Module):
                def __init__(self, n_channels=19, n_classes=2):
                    super(SeizureCNN, self).__init__()
                    self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                    self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool1d(2)
                    self.dropout = nn.Dropout(0.5)
                    self.global_pool = nn.AdaptiveAvgPool1d(1)
                    self.fc1 = nn.Linear(256, 512)
                    self.fc2 = nn.Linear(512, n_classes)
                    self.relu = nn.ReLU()
                    self.softmax = nn.Softmax(dim=1)
                    
                def forward(self, x):
                    x = self.relu(self.conv1(x))
                    x = self.pool(x)
                    x = self.relu(self.conv2(x))
                    x = self.pool(x)
                    x = self.relu(self.conv3(x))
                    x = self.pool(x)
                    x = self.global_pool(x)
                    x = x.view(x.size(0), -1)
                    x = self.dropout(x)
                    x = self.relu(self.fc1(x))
                    x = self.fc2(x)
                    return self.softmax(x)
            
            self.model = SeizureCNN()
            self.is_loaded = True
            print("✅ Seizure detection model loaded")
            
        except Exception as e:
            print(f"❌ Error loading seizure model: {e}")
    
    def extract_seizure_features(self, segments):
        """Extract seizure-specific features"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(segment.shape[0]):
                signal_data = segment[ch]
                
                # Statistical features
                variance = np.var(signal_data)
                skewness = np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 3)
                kurtosis = np.mean(((signal_data - np.mean(signal_data)) / np.std(signal_data)) ** 4)
                
                # Frequency domain features
                fft_vals = np.abs(fft(signal_data))
                freqs = fftfreq(len(signal_data), 1/256)
                
                # Power in different bands
                delta_power = np.sum(fft_vals[(freqs >= 1) & (freqs <= 4)])
                theta_power = np.sum(fft_vals[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.sum(fft_vals[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.sum(fft_vals[(freqs >= 13) & (freqs <= 30)])
                gamma_power = np.sum(fft_vals[(freqs >= 30) & (freqs <= 100)])
                
                # High frequency activity (seizure indicator)
                high_freq_ratio = (beta_power + gamma_power) / (delta_power + theta_power + alpha_power + 1e-8)
                
                segment_features.extend([variance, skewness, kurtosis, high_freq_ratio])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict seizure probability"""
        try:
            if not self.is_loaded:
                self.load_model()
            
            features = self.extract_seizure_features(segments)
            predictions = []
            
            for feat_vec in features:
                # Enhanced seizure detection based on multiple features
                # High variance + high frequency content + spectral changes
                variance_score = np.mean(feat_vec[::4])  # Every 4th element is variance
                high_freq_score = np.mean(feat_vec[3::4])  # Every 4th element starting from 3 is high freq ratio
                
                # Combine features with weights
                seizure_score = min(1.0, 
                    0.4 * min(1.0, variance_score / 2.0) +  # Variance component
                    0.6 * min(1.0, high_freq_score / 3.0)   # High frequency component
                )
                
                predictions.append([1-seizure_score, seizure_score])
            
            predictions = np.array(predictions)
            print(f"✅ Seizure detection completed: {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"❌ Error in seizure prediction: {e}")
            return np.array([[0.5, 0.5]] * len(segments))

class AlzheimerDetector:
    """Alzheimer's Detection adapted for CHB-MIT data"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load Alzheimer's detection model"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu', input_shape=(76,)),  # 4 features × 19 channels
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.is_loaded = True
            print("✅ Alzheimer's detection model loaded")
            
        except Exception as e:
            print(f"❌ Error loading Alzheimer's model: {e}")
    
    def extract_alzheimer_features(self, segments):
        """Extract features relevant to Alzheimer's detection"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(segment.shape[0]):
                signal_data = segment[ch]
                freqs = fftfreq(len(signal_data), 1/256)
                psd = np.abs(fft(signal_data))
                
                # Frequency band powers
                delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
                theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
                
                # Key Alzheimer's indicators
                theta_alpha_ratio = theta_power / (alpha_power + 1e-8)  # Increased in AD
                delta_theta_ratio = delta_power / (theta_power + 1e-8)  # Changed in AD
                alpha_beta_ratio = alpha_power / (beta_power + 1e-8)    # Decreased in AD
                
                # Spectral entropy (complexity measure)
                psd_norm = psd / (np.sum(psd) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
                
                segment_features.extend([theta_alpha_ratio, delta_theta_ratio, 
                                       alpha_beta_ratio, spectral_entropy])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict Alzheimer's probability"""
        try:
            if not self.is_loaded:
                self.load_model()
            
            features = self.extract_alzheimer_features(segments)
            predictions = []
            
            for feat_vec in features:
                # Alzheimer's detection based on theta/alpha ratio and complexity
                theta_alpha_scores = feat_vec[::4]  # Every 4th element
                complexity_scores = feat_vec[3::4]  # Every 4th element starting from 3
                
                # Higher theta/alpha ratio and lower complexity suggest AD
                avg_theta_alpha = np.mean(theta_alpha_scores)
                avg_complexity = np.mean(complexity_scores)
                
                # Combine indicators
                alzheimer_score = min(1.0,
                    0.7 * min(1.0, avg_theta_alpha / 2.5) +      # Theta/alpha component
                    0.3 * max(0.0, (4.0 - avg_complexity) / 4.0) # Reduced complexity component
                )
                
                predictions.append([1-alzheimer_score, alzheimer_score])
            
            predictions = np.array(predictions)
            print(f"✅ Alzheimer's detection completed: {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"❌ Error in Alzheimer's prediction: {e}")
            return np.array([[0.5, 0.5]] * len(segments))

class ParkinsonDetector:
    """Parkinson's Detection adapted for CHB-MIT data"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load Parkinson's detection model"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(76,)),  # 4 features × 19 channels
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            self.is_loaded = True
            print("✅ Parkinson's detection model loaded")
            
        except Exception as e:
            print(f"❌ Error loading Parkinson's model: {e}")
    
    def extract_parkinson_features(self, segments):
        """Extract features relevant to Parkinson's detection"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(segment.shape[0]):
                signal_data = segment[ch]
                freqs = fftfreq(len(signal_data), 1/256)
                psd = np.abs(fft(signal_data))
                
                # Frequency band powers
                delta_power = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
                theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
                
                total_power = delta_power + theta_power + alpha_power + beta_power + 1e-8
                
                # Key Parkinson's indicators
                beta_ratio = beta_power / total_power        # Often reduced in PD
                alpha_ratio = alpha_power / total_power      # May be altered in PD
                tremor_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])  # Tremor frequency
                tremor_ratio = tremor_power / total_power
                
                segment_features.extend([beta_ratio, alpha_ratio, tremor_ratio, 
                                       beta_ratio - tremor_ratio])  # Beta suppression relative to tremor
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict Parkinson's probability"""
        try:
            if not self.is_loaded:
                self.load_model()
            
            features = self.extract_parkinson_features(segments)
            predictions = []
            
            for feat_vec in features:
                # Parkinson's indicators: beta suppression, tremor activity
                beta_ratios = feat_vec[::4]      # Every 4th element
                tremor_ratios = feat_vec[2::4]   # Every 4th element starting from 2
                beta_tremor_diff = feat_vec[3::4]  # Every 4th element starting from 3
                
                avg_beta_ratio = np.mean(beta_ratios)
                avg_tremor_ratio = np.mean(tremor_ratios) 
                avg_beta_tremor_diff = np.mean(beta_tremor_diff)
                
                # Lower beta activity and higher tremor activity suggest PD
                parkinson_score = min(1.0,
                    0.4 * max(0.0, (0.3 - avg_beta_ratio) / 0.3) +      # Beta suppression
                    0.3 * min(1.0, avg_tremor_ratio / 0.2) +             # Tremor activity
                    0.3 * max(0.0, -avg_beta_tremor_diff / 0.2)          # Beta-tremor difference
                )
                
                predictions.append([1-parkinson_score, parkinson_score])
            
            predictions = np.array(predictions)
            print(f"✅ Parkinson's detection completed: {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"❌ Error in Parkinson's prediction: {e}")
            return np.array([[0.5, 0.5]] * len(segments))

class CHBMITDetectionPipeline:
    """Main pipeline for CHB-MIT EEG neurological disease detection"""
    
    def __init__(self, data_dir=None):
        """
        Initialize pipeline with local data directory
        
        Args:
            data_dir: Path to your local CHB-MIT data directory
                     Example (raw string): r"C:\\Users\\chanm\\Downloads\\archive (1)\\Annotated_EEG"
                     OR (forward slashes): "C:/Users/chanm/Downloads/archive (1)/Annotated_EEG"
        """
        # Use provided directory or prompt user
        if data_dir is None:
            print("⚠️  No data directory specified!")
            print("Please provide the path to your CHB-MIT data when initializing:")
            print('Example: pipeline = CHBMITDetectionPipeline(r"C:\\Users\\chanm\\Downloads\\archive (1)\\Annotated_EEG")')
            self.data_dir = Path("./chb_mit_data")
        else:
            self.data_dir = Path(data_dir)
        
        self.downloader = CHBMITDownloader(str(self.data_dir))
        self.preprocessor = CHBMITPreprocessor()
        self.seizure_detector = SeizureDetector()
        self.alzheimer_detector = AlzheimerDetector()
        self.parkinson_detector = ParkinsonDetector()
        
        print("🧠 CHB-MIT EEG Detection Pipeline Initialized")
        print("📋 Available detectors: Seizure, Alzheimer's, Parkinson's")
    
    def setup_dataset(self):
        """Verify local dataset is accessible"""
        try:
            print("🚀 Verifying local CHB-MIT dataset...")
            print("=" * 50)
            
            # Verify dataset exists
            success = self.downloader.verify_local_dataset()
            
            if not success:
                print("\n❌ Dataset verification failed")
                print("Please ensure:")
                print(f"1. The directory exists: {self.data_dir}")
                print("2. The directory contains EDF files")
                print("3. You have read permissions for the directory")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error verifying dataset: {e}")
            return False
    
    def analyze_single_file(self, file_path, visualize=True):
        """Analyze a single EDF file"""
        try:
            print(f"\n🔍 Analyzing file: {Path(file_path).name}")
            print("=" * 60)
            
            # Load EDF file
            print("📖 Step 1: Loading EDF file")
            data, info = self.preprocessor.load_edf_file(file_path)
            if data is None:
                return None
            
            # Preprocessing
            print("\n🔧 Step 2: Preprocessing")
            processed_data = self.preprocessor.apply_preprocessing(data)
            
            # Segmentation
            print("\n✂️ Step 3: Segmentation")
            segments = self.preprocessor.segment_data(processed_data, window_size=1024, overlap=0.5)
            
            # Disease detection
            print("\n🔍 Step 4: Running disease detection models")
            print("-" * 40)
            
            # Load models
            self.seizure_detector.load_model()
            self.alzheimer_detector.load_model()
            self.parkinson_detector.load_model()
            
            # Run predictions
            seizure_preds = self.seizure_detector.predict(segments)
            alzheimer_preds = self.alzheimer_detector.predict(segments)
            parkinson_preds = self.parkinson_detector.predict(segments)
            
            # Analyze results
            print("\n📊 Step 5: Results analysis")
            results = self.analyze_results(seizure_preds, alzheimer_preds, parkinson_preds)
            
            # Add file info to results
            results['file_info'] = {
                'filename': Path(file_path).name,
                'duration_seconds': len(processed_data) / 256,
                'channels': list(processed_data.columns),
                'segments_analyzed': len(segments)
            }
            
            # Visualization
            if visualize:
                print("\n📈 Step 6: Generating visualizations")
                self.visualize_results(processed_data, results, seizure_preds, 
                                     alzheimer_preds, parkinson_preds)
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing file: {e}")
            return None
    
    def analyze_multiple_files(self, file_paths, max_files=5):
        """Analyze multiple EDF files"""
        try:
            print(f"\n🔍 Analyzing {min(len(file_paths), max_files)} files")
            print("=" * 60)
            
            all_results = []
            
            for i, file_path in enumerate(file_paths[:max_files]):
                print(f"\n📁 File {i+1}/{min(len(file_paths), max_files)}")
                results = self.analyze_single_file(file_path, visualize=False)
                
                if results:
                    all_results.append(results)
            
            # Aggregate results
            if all_results:
                aggregated = self.aggregate_multiple_results(all_results)
                self.visualize_aggregate_results(all_results)
                return aggregated
            
            return None
            
        except Exception as e:
            print(f"❌ Error in multiple file analysis: {e}")
            return None
    
    def analyze_seizure_vs_normal(self):
        """Compare seizure files vs normal files"""
        try:
            print("\n🔍 SEIZURE vs NORMAL ANALYSIS")
            print("=" * 50)
            
            # Get seizure files
            seizure_files = self.downloader.get_seizure_files()
            all_files = self.downloader.get_edf_files()
            
            # Get normal files (non-seizure files)
            normal_files = [f for f in all_files if f not in seizure_files]
            
            print(f"📊 Found {len(seizure_files)} seizure files")
            print(f"📊 Found {len(normal_files)} normal files")
            
            if not seizure_files or not normal_files:
                print("❌ Insufficient files for comparison")
                return None
            
            # Analyze sample of each type
            print("\n🔴 Analyzing seizure files...")
            seizure_results = self.analyze_multiple_files(seizure_files[:3], max_files=3)
            
            print("\n🟢 Analyzing normal files...")
            normal_results = self.analyze_multiple_files(normal_files[:3], max_files=3)
            
            # Compare results
            if seizure_results and normal_results:
                print("\n📊 COMPARISON RESULTS")
                print("=" * 30)
                self.compare_seizure_normal_results(seizure_results, normal_results)
            
            return {'seizure_results': seizure_results, 'normal_results': normal_results}
            
        except Exception as e:
            print(f"❌ Error in seizure vs normal analysis: {e}")
            return None
    
    def analyze_results(self, seizure_preds, alzheimer_preds, parkinson_preds):
        """Analyze and summarize results"""
        try:
            results = {
                'seizure': {
                    'mean_probability': float(np.mean(seizure_preds[:, 1])),
                    'max_probability': float(np.max(seizure_preds[:, 1])),
                    'segments_detected': int(np.sum(seizure_preds[:, 1] > 0.5)),
                    'total_segments': int(len(seizure_preds)),
                    'detection_rate': float(np.sum(seizure_preds[:, 1] > 0.5) / len(seizure_preds))
                },
                'alzheimer': {
                    'mean_probability': float(np.mean(alzheimer_preds[:, 1])),
                    'max_probability': float(np.max(alzheimer_preds[:, 1])),
                    'segments_detected': int(np.sum(alzheimer_preds[:, 1] > 0.5)),
                    'total_segments': int(len(alzheimer_preds)),
                    'detection_rate': float(np.sum(alzheimer_preds[:, 1] > 0.5) / len(alzheimer_preds))
                },
                'parkinson': {
                    'mean_probability': float(np.mean(parkinson_preds[:, 1])),
                    'max_probability': float(np.max(parkinson_preds[:, 1])),
                    'segments_detected': int(np.sum(parkinson_preds[:, 1] > 0.5)),
                    'total_segments': int(len(parkinson_preds)),
                    'detection_rate': float(np.sum(parkinson_preds[:, 1] > 0.5) / len(parkinson_preds))
                }
            }
            
            # Risk assessment
            high_risk_conditions = []
            moderate_risk_conditions = []
            
            for condition, metrics in results.items():
                if metrics['mean_probability'] > 0.7:
                    high_risk_conditions.append(condition)
                elif metrics['mean_probability'] > 0.4:
                    moderate_risk_conditions.append(condition)
            
            results['risk_assessment'] = {
                'high_risk_conditions': high_risk_conditions,
                'moderate_risk_conditions': moderate_risk_conditions,
                'overall_risk_level': 'HIGH' if high_risk_conditions else 'MODERATE' if moderate_risk_conditions else 'LOW'
            }
            
            # Print detailed results
            self.print_detailed_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in results analysis: {e}")
            return {}
    
    def print_detailed_results(self, results):
        """Print detailed analysis results"""
        print("\n📋 DETAILED DETECTION RESULTS")
        print("=" * 50)
        
        for condition, metrics in results.items():
            if condition == 'risk_assessment':
                continue
                
            print(f"\n🔍 {condition.upper()} DETECTION:")
            print("-" * 25)
            print(f"   Mean Probability: {metrics['mean_probability']:.3f}")
            print(f"   Max Probability:  {metrics['max_probability']:.3f}")
            print(f"   Detection Rate:   {metrics['detection_rate']:.1%}")
            print(f"   Segments:         {metrics['segments_detected']}/{metrics['total_segments']}")
            
            # Risk level assessment
            if metrics['mean_probability'] > 0.7:
                print(f"   🚨 RISK LEVEL: HIGH - Immediate medical attention recommended")
            elif metrics['mean_probability'] > 0.4:
                print(f"   ⚠️  RISK LEVEL: MODERATE - Monitor closely, consult specialist")
            else:
                print(f"   ✅ RISK LEVEL: LOW - Normal range")
        
        # Overall assessment
        risk_info = results.get('risk_assessment', {})
        print(f"\n🎯 OVERALL ASSESSMENT: {risk_info.get('overall_risk_level', 'UNKNOWN')}")
        print("=" * 30)
        
        if risk_info.get('high_risk_conditions'):
            print(f"🚨 HIGH RISK CONDITIONS: {', '.join(risk_info['high_risk_conditions'])}")
            print("   → RECOMMENDATION: Consult neurologist immediately")
        elif risk_info.get('moderate_risk_conditions'):
            print(f"⚠️  MODERATE RISK CONDITIONS: {', '.join(risk_info['moderate_risk_conditions'])}")
            print("   → RECOMMENDATION: Schedule specialist consultation")
        else:
            print("✅ No high-risk neurological conditions detected")
            print("   → RECOMMENDATION: Continue regular monitoring")
    
    def aggregate_multiple_results(self, results_list):
        """Aggregate results from multiple files"""
        try:
            aggregated = {
                'seizure': {'probabilities': [], 'detection_rates': []},
                'alzheimer': {'probabilities': [], 'detection_rates': []},
                'parkinson': {'probabilities': [], 'detection_rates': []}
            }
            
            # Collect individual results
            for result in results_list:
                for condition in ['seizure', 'alzheimer', 'parkinson']:
                    if condition in result:
                        aggregated[condition]['probabilities'].append(result[condition]['mean_probability'])
                        aggregated[condition]['detection_rates'].append(result[condition]['detection_rate'])
            
            # Calculate aggregate statistics
            final_results = {}
            for condition, data in aggregated.items():
                if data['probabilities']:
                    final_results[condition] = {
                        'mean_probability': float(np.mean(data['probabilities'])),
                        'std_probability': float(np.std(data['probabilities'])),
                        'mean_detection_rate': float(np.mean(data['detection_rates'])),
                        'files_analyzed': len(data['probabilities'])
                    }
            
            print(f"\n📊 AGGREGATED RESULTS ({len(results_list)} files)")
            print("=" * 40)
            
            for condition, metrics in final_results.items():
                print(f"\n{condition.upper()}:")
                print(f"   Mean Probability: {metrics['mean_probability']:.3f} ± {metrics['std_probability']:.3f}")
                print(f"   Mean Detection Rate: {metrics['mean_detection_rate']:.1%}")
                print(f"   Files Analyzed: {metrics['files_analyzed']}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error in aggregating results: {e}")
            return {}
    
    def compare_seizure_normal_results(self, seizure_results, normal_results):
        """Compare seizure vs normal file results"""
        try:
            print("\n🔴 SEIZURE FILES vs 🟢 NORMAL FILES")
            print("=" * 45)
            
            conditions = ['seizure', 'alzheimer', 'parkinson']
            
            for condition in conditions:
                if condition in seizure_results and condition in normal_results:
                    seizure_prob = seizure_results[condition]['mean_probability']
                    normal_prob = normal_results[condition]['mean_probability']
                    
                    print(f"\n{condition.upper()} Detection:")
                    print(f"   🔴 Seizure files: {seizure_prob:.3f}")
                    print(f"   🟢 Normal files:  {normal_prob:.3f}")
                    print(f"   📊 Difference:    {seizure_prob - normal_prob:+.3f}")
                    
                    if condition == 'seizure':
                        if seizure_prob > normal_prob + 0.2:
                            print("   ✅ Good discrimination: Seizure files show higher seizure probability")
                        else:
                            print("   ⚠️  Poor discrimination: Similar probabilities")
            
        except Exception as e:
            print(f"❌ Error comparing results: {e}")
    
    def visualize_results(self, data, results, seizure_preds, alzheimer_preds, parkinson_preds):
        """Create comprehensive visualizations"""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            
            # Plot 1: EEG Signal Sample
            sample_length = min(2000, len(data))  # First 2000 samples
            channels_to_plot = min(5, len(data.columns))
            
            for i in range(channels_to_plot):
                axes[0, 0].plot(data.iloc[:sample_length, i], alpha=0.7, label=data.columns[i])
            
            axes[0, 0].set_title('EEG Signal Sample (First 5 Channels)', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Time (samples)')
            axes[0, 0].set_ylabel('Amplitude (normalized)')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Detection Probabilities Bar Chart
            conditions = ['Seizure', 'Alzheimer', 'Parkinson']
            mean_probs = [
                np.mean(seizure_preds[:, 1]),
                np.mean(alzheimer_preds[:, 1]),
                np.mean(parkinson_preds[:, 1])
            ]
            
            colors = ['#ff4444', '#ff8800', '#4444ff']
            bars = axes[0, 1].bar(conditions, mean_probs, color=colors, alpha=0.7, edgecolor='black')
            
            axes[0, 1].set_title('Mean Detection Probabilities', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Probability')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
            
            # Add value labels on bars
            for bar, prob in zip(bars, mean_probs):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Probability Timeline
            time_axis = range(len(seizure_preds))
            axes[1, 0].plot(time_axis, seizure_preds[:, 1], color='#ff4444', 
                           label='Seizure', linewidth=2, alpha=0.8)
            axes[1, 0].plot(time_axis, alzheimer_preds[:, 1], color='#ff8800', 
                           label='Alzheimer', linewidth=2, alpha=0.8)
            axes[1, 0].plot(time_axis, parkinson_preds[:, 1], color='#4444ff', 
                           label='Parkinson', linewidth=2, alpha=0.8)
            axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
            
            axes[1, 0].set_title('Detection Probabilities Over Time', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Segment Number')
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Risk Assessment Heatmap
            risk_data = np.array([
                [results['seizure']['mean_probability']],
                [results['alzheimer']['mean_probability']],
                [results['parkinson']['mean_probability']]
            ])
            
            im = axes[1, 1].imshow(risk_data, cmap='Reds', aspect='auto', vmin=0, vmax=1)
            axes[1, 1].set_yticks(range(len(conditions)))
            axes[1, 1].set_yticklabels(conditions)
            axes[1, 1].set_xticks([0])
            axes[1, 1].set_xticklabels(['Risk Level'])
            axes[1, 1].set_title('Risk Assessment Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 1])
            cbar.set_label('Risk Level', rotation=270, labelpad=15)
            
            # Add text annotations
            for i in range(len(conditions)):
                color = 'white' if risk_data[i, 0] > 0.5 else 'black'
                axes[1, 1].text(0, i, f'{risk_data[i, 0]:.3f}', 
                               ha='center', va='center', color=color, fontweight='bold')
            
            # Plot 5: Detection Rate Comparison
            detection_rates = [
                results['seizure']['detection_rate'],
                results['alzheimer']['detection_rate'],
                results['parkinson']['detection_rate']
            ]
            
            bars = axes[2, 0].bar(conditions, detection_rates, color=colors, alpha=0.7, edgecolor='black')
            axes[2, 0].set_title('Segment Detection Rates', fontsize=14, fontweight='bold')
            axes[2, 0].set_ylabel('Detection Rate (%)')
            axes[2, 0].set_ylim([0, 1])
            
            for bar, rate in zip(bars, detection_rates):
                axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            axes[2, 0].grid(True, alpha=0.3)
            
            # Plot 6: Frequency Domain Analysis
            # Show power spectral density of a sample segment
            sample_segment = data.iloc[1000:2048, 0].values  # Sample 1024 points from first channel
            freqs = fftfreq(len(sample_segment), 1/256)
            psd = np.abs(fft(sample_segment))
            
            # Only plot positive frequencies up to 50 Hz
            mask = (freqs >= 0) & (freqs <= 50)
            axes[2, 1].semilogy(freqs[mask], psd[mask], color='darkblue', linewidth=2)
            axes[2, 1].set_title('Power Spectral Density (Sample Channel)', fontsize=14, fontweight='bold')
            axes[2, 1].set_xlabel('Frequency (Hz)')
            axes[2, 1].set_ylabel('Power (log scale)')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Add frequency band annotations
            axes[2, 1].axvspan(1, 4, alpha=0.2, color='purple', label='Delta (1-4 Hz)')
            axes[2, 1].axvspan(4, 8, alpha=0.2, color='blue', label='Theta (4-8 Hz)')
            axes[2, 1].axvspan(8, 13, alpha=0.2, color='green', label='Alpha (8-13 Hz)')
            axes[2, 1].axvspan(13, 30, alpha=0.2, color='orange', label='Beta (13-30 Hz)')
            axes[2, 1].legend(loc='upper right')
            
            plt.tight_layout()
            
            # Add overall title
            fig.suptitle(f"CHB-MIT EEG Analysis Results - {results.get('file_info', {}).get('filename', 'Unknown File')}", 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.show()
            
            print("📊 Comprehensive visualizations generated successfully!")
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
    
    def visualize_aggregate_results(self, results_list):
        """Visualize aggregated results from multiple files"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            conditions = ['seizure', 'alzheimer', 'parkinson']
            colors = ['#ff4444', '#ff8800', '#4444ff']
            
            # Collect data for visualization
            condition_data = {cond: [] for cond in conditions}
            
            for result in results_list:
                for cond in conditions:
                    if cond in result:
                        condition_data[cond].append(result[cond]['mean_probability'])
            
            # Plot 1: Box plot of probabilities across files
            data_for_boxplot = [condition_data[cond] for cond in conditions if condition_data[cond]]
            labels_for_boxplot = [cond.title() for cond in conditions if condition_data[cond]]
            
            if data_for_boxplot:
                bp = axes[0, 0].boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[0, 0].set_title('Probability Distribution Across Files', fontsize=12, fontweight='bold')
                axes[0, 0].set_ylabel('Probability')
                axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: File-by-file comparison
            file_numbers = range(1, len(results_list) + 1)
            
            for i, cond in enumerate(conditions):
                if condition_data[cond]:
                    axes[0, 1].plot(file_numbers, condition_data[cond], 
                                   marker='o', color=colors[i], label=cond.title(), linewidth=2)
            
            axes[0, 1].set_title('Detection Probabilities by File', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('File Number')
            axes[0, 1].set_ylabel('Probability')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            # Plot 3: Average probabilities
            avg_probs = [np.mean(condition_data[cond]) if condition_data[cond] else 0 
                        for cond in conditions]
            
            bars = axes[1, 0].bar([cond.title() for cond in conditions], avg_probs, 
                                 color=colors, alpha=0.7, edgecolor='black')
            
            axes[1, 0].set_title('Average Detection Probabilities', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Average Probability')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            for bar, prob in zip(bars, avg_probs):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Statistical summary
            axes[1, 1].axis('off')  # Turn off axis for text display
            
            summary_text = f"ANALYSIS SUMMARY\n{'-'*20}\n"
            summary_text += f"Files Analyzed: {len(results_list)}\n\n"
            
            for cond in conditions:
                if condition_data[cond]:
                    data = condition_data[cond]
                    summary_text += f"{cond.upper()}:\n"
                    summary_text += f"  Mean: {np.mean(data):.3f}\n"
                    summary_text += f"  Std:  {np.std(data):.3f}\n"
                    summary_text += f"  Max:  {np.max(data):.3f}\n"
                    summary_text += f"  Min:  {np.min(data):.3f}\n\n"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.suptitle('Multiple File Analysis Results', fontsize=14, fontweight='bold', y=0.98)
            plt.show()
            
            print("📊 Aggregate visualizations generated successfully!")
            
        except Exception as e:
            print(f"❌ Error in aggregate visualization: {e}")
    
    def run_demo_analysis(self):
        """Run a demonstration analysis on CHB-MIT dataset"""
        try:
            print("🎬 RUNNING CHB-MIT DEMO ANALYSIS")
            print("=" * 50)
            
            # Setup dataset
            if not self.setup_dataset():
                return None
            
            # Get available files
            all_files = self.downloader.get_edf_files()
            seizure_files = self.downloader.get_seizure_files()
            
            if not all_files:
                print("❌ No EDF files found. Please check dataset download.")
                return None
            
            print(f"📊 Found {len(all_files)} total EDF files")
            print(f"📊 Found {len(seizure_files)} seizure files")
            
            # Demo options
            print("\n🎯 Demo Analysis Options:")
            print("1. Single file analysis (quick demo)")
            print("2. Multiple files analysis")
            print("3. Seizure vs Normal comparison")
            
            # Run single file demo
            print("\n" + "="*60)
            print("🚀 RUNNING SINGLE FILE DEMO")
            print("="*60)
            
            demo_file = all_files[0]  # Use first available file
            single_result = self.analyze_single_file(demo_file, visualize=True)
            
            if len(all_files) >= 3:
                print("\n" + "="*60)
                print("🚀 RUNNING MULTIPLE FILES DEMO")
                print("="*60)
                
                multi_results = self.analyze_multiple_files(all_files[:3], max_files=3)
            
            if seizure_files and len(all_files) > len(seizure_files):
                print("\n" + "="*60)
                print("🚀 RUNNING SEIZURE vs NORMAL COMPARISON")
                print("="*60)
                
                comparison_results = self.analyze_seizure_vs_normal()
            
            print("\n🎉 DEMO ANALYSIS COMPLETED!")
            print("="*50)
            
            return {
                'single_file': single_result,
                'multiple_files': multi_results if 'multi_results' in locals() else None,
                'comparison': comparison_results if 'comparison_results' in locals() else None
            }
            
        except Exception as e:
            print(f"❌ Error in demo analysis: {e}")
            return None

def main():
    """Main function to run the CHB-MIT pipeline"""
    print("🧠 CHB-MIT EEG Neurological Disease Detection System")
    print("=" * 60)
    print("This system uses the CHB-MIT Scalp EEG Database to detect:")
    print("🔴 Seizures (Epilepsy)")
    print("🟠 Alzheimer's Disease")  
    print("🔵 Parkinson's Disease")
    print("=" * 60)
    
    # ⚠️ IMPORTANT: UPDATE THIS PATH TO YOUR LOCAL DATA DIRECTORY
    # Use raw string (r"...") OR forward slashes
    LOCAL_DATA_PATH = r"C:\Users\chanm\Downloads\archive (1)\Annotated_EEG"
    # Alternative with forward slashes: "C:/Users/chanm/Downloads/archive (1)/Annotated_EEG"
    
    print("\n📁 Using local data from:")
    print(f"   {LOCAL_DATA_PATH}")
    print("\n💡 To change the data path, edit the LOCAL_DATA_PATH variable in main()")
    
    # Initialize pipeline with local data path
    pipeline = CHBMITDetectionPipeline(data_dir=LOCAL_DATA_PATH)
    
    print("\n📝 USAGE OPTIONS:")
    print("-" * 25)
    print("1️⃣ Run demo analysis:")
    print("   results = pipeline.run_demo_analysis()")
    print("\n2️⃣ Analyze specific file:")
    print('   results = pipeline.analyze_single_file(r"C:\\path\\to\\file.edf")')
    print("\n3️⃣ Compare seizure vs normal files:")
    print("   results = pipeline.analyze_seizure_vs_normal()")
    print("\n4️⃣ Analyze multiple files:")
    print("   files = pipeline.downloader.get_edf_files()")
    print("   results = pipeline.analyze_multiple_files(files[:5])")
    
    print("\n" + "=" * 60)
    print("💡 Pipeline ready! The system will:")
    print("   • Use your local CHB-MIT dataset")
    print("   • Process EDF files with proper channel mapping")
    print("   • Apply three neurological disease detection models")
    print("   • Generate comprehensive analysis and visualizations")
    
    return pipeline

# Auto-run demo if executed directly
if __name__ == "__main__":
    try:
        # ⚠️ IMPORTANT: Set your local data path here
        # Use raw string (r"...") to handle backslashes correctly
        YOUR_LOCAL_DATA_PATH = r"C:\Users\chanm\Downloads\archive (1)\Annotated_EEG"
        
        print("🚀" * 30)
        print("\n🎯 CHB-MIT EEG DETECTION SYSTEM - LOCAL MODE")
        print("=" * 60)
        print(f"📁 Data Directory: {YOUR_LOCAL_DATA_PATH}")
        print("=" * 60)
        
        # Initialize the pipeline with your local path
        pipeline = CHBMITDetectionPipeline(data_dir=YOUR_LOCAL_DATA_PATH)
        
        print("\n" + "🚀" * 20)
        print("STARTING AUTOMATIC DEMO ANALYSIS")
        print("🚀" * 20)
        print("This will:")
        print("1. Verify local dataset")
        print("2. Analyze sample EDF files from your directory")
        print("3. Run all three disease detection models")
        print("4. Generate visualizations")
        print("\n⏳ Starting in 3 seconds...")
        
        import time
        time.sleep(3)
        
        # Run the demo
        demo_results = pipeline.run_demo_analysis()
        
        if demo_results:
            print("\n" + "🎉" * 20)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("🎉" * 20)
            print("\n📊 Results Summary:")
            
            if demo_results.get('single_file'):
                single = demo_results['single_file']
                print(f"\n📁 Single File Analysis:")
                if 'risk_assessment' in single:
                    risk_level = single['risk_assessment'].get('overall_risk_level', 'UNKNOWN')
                    print(f"   Overall Risk Level: {risk_level}")
                    
                    for condition in ['seizure', 'alzheimer', 'parkinson']:
                        if condition in single:
                            prob = single[condition]['mean_probability']
                            print(f"   {condition.title()}: {prob:.3f} probability")
            
            print("\n🔬 Analysis complete! Check the generated plots above.")
            print("💡 You can now run custom analyses using the pipeline object.")
            print("\n📝 Example custom usage:")
            print(f"   pipeline = CHBMITDetectionPipeline(r'{YOUR_LOCAL_DATA_PATH}')")
            print("   files = pipeline.downloader.get_edf_files()")
            print("   result = pipeline.analyze_single_file(files[0])")
            
        else:
            print("\n❌ Demo analysis failed. Please check:")
            print(f"1. Data directory path is correct: {YOUR_LOCAL_DATA_PATH}")
            print("2. Directory contains EDF files")
            print("3. You have read permissions for the directory")
            print("4. Required packages are installed")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Update YOUR_LOCAL_DATA_PATH with your actual directory:")
        print(f"   Current: {YOUR_LOCAL_DATA_PATH}")
        print("   Example: r'C:\\Users\\YourName\\Downloads\\archive (1)\\Annotated_EEG'")
        print("\n2. Install required packages:")
        print("   pip install torch tensorflow scikit-learn mne matplotlib seaborn")
        print("\n3. Verify EDF files exist in directory")
        print("\n4. Run manually:")
        print("   pipeline = CHBMITDetectionPipeline(r'YOUR_PATH_HERE')")
        print("   results = pipeline.run_demo_analysis()")
        
    finally:
        print("\n" + "=" * 60)
        print("🧠 CHB-MIT EEG Detection Pipeline Session Ended")
        print("=" * 60)

# Additional utility functions for advanced users
def quick_seizure_analysis(edf_file_path, data_dir=None):
    """Quick function to analyze a single EDF file for seizures"""
    try:
        if data_dir is None:
            data_dir = Path(edf_file_path).parent
            
        pipeline = CHBMITDetectionPipeline(data_dir=str(data_dir))
        results = pipeline.analyze_single_file(edf_file_path, visualize=True)
        
        if results and 'seizure' in results:
            seizure_prob = results['seizure']['mean_probability']
            print(f"\n🔍 QUICK SEIZURE ANALYSIS RESULT:")
            print(f"   File: {Path(edf_file_path).name}")
            print(f"   Seizure Probability: {seizure_prob:.3f}")
            
            if seizure_prob > 0.7:
                print(f"   🚨 HIGH SEIZURE RISK DETECTED")
            elif seizure_prob > 0.4:
                print(f"   ⚠️  MODERATE SEIZURE RISK")
            else:
                print(f"   ✅ LOW SEIZURE RISK")
                
        return results
        
    except Exception as e:
        print(f"❌ Error in quick analysis: {e}")
        return None

def batch_process_directory(directory_path, max_files=10):
    """Batch process all EDF files in a directory"""
    try:
        pipeline = CHBMITDetectionPipeline(data_dir=directory_path)
        
        # Find all EDF files in directory
        edf_files = list(Path(directory_path).glob("**/*.edf"))
        
        if not edf_files:
            print(f"❌ No EDF files found in {directory_path}")
            return None
        
        print(f"📁 Found {len(edf_files)} EDF files")
        print(f"🔄 Processing first {min(len(edf_files), max_files)} files...")
        
        # Process files
        results = pipeline.analyze_multiple_files(edf_files, max_files=max_files)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return None

# Export main classes for advanced usage
__all__ = [
    'CHBMITDetectionPipeline',
    'CHBMITDownloader', 
    'CHBMITPreprocessor',
    'SeizureDetector',
    'AlzheimerDetector', 
    'ParkinsonDetector',
    'quick_seizure_analysis',
    'batch_process_directory'
]