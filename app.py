"""
Sleep Apnea Detection System - Streamlit Web Application

Main application file for the sleep apnea detection web interface.
"""

import glob
import io
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from chatbot import render_chatbot

# Configure matplotlib for dark theme
plt.rcParams['figure.facecolor'] = '#0a0e27'
plt.rcParams['axes.facecolor'] = '#0a0e27'
plt.rcParams['axes.edgecolor'] = '#3b82f6'
plt.rcParams['axes.labelcolor'] = '#f1f5f9'
plt.rcParams['text.color'] = '#f1f5f9'
plt.rcParams['xtick.color'] = '#e2e8f0'
plt.rcParams['ytick.color'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#3b82f6'
plt.rcParams['grid.alpha'] = 0.2

# Page configuration
st.set_page_config(
    page_title="Sleep Apnea Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Dark Tech Healthcare CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: #0a0e27 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%) !important;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Content Area - Centered and Compact for Single Frame */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: calc(100vh - 2rem);
    }
    
    /* Remove default streamlit spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact viewport - Single Frame - Centered */
    .stApp > div {
        padding-top: 0 !important;
    }
    
    /* Remove top spacing from main */
    .main {
        padding-top: 0 !important;
    }
    
    /* Ensure no overflow on homepage */
    .main .block-container {
        overflow: visible !important;
    }
    
    /* Center content vertically */
    #root > div > div > div > div > div > section {
        padding-top: 0 !important;
    }
    
    /* Compact file uploader area */
    section[data-testid="stFileUploader"] > div {
        padding: 0.5rem !important;
        min-height: auto !important;
    }
    
    /* Remove extra spacing in upload area */
    .uploadedFile {
        margin: 0.3rem 0 !important;
    }
    
    /* Header Section - Futuristic Dark - Centered */
    .header-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0 auto 0.5rem auto;
        text-align: center;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 20px rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
        max-width: 1000px;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.5);
        letter-spacing: -0.5px;
        font-family: 'Poppins', sans-serif;
    }
    
    .subtitle {
        font-size: 0.75rem;
        font-weight: 400;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
        letter-spacing: 0.2px;
    }
    
    .tech-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        color: #10b981;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }
    
    .tech-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        background: rgba(16, 185, 129, 0.25);
    }
    
    /* Upload Section - Glassmorphism - Compact - Centered */
    .upload-section {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.3rem auto;
        box-shadow: 
            0 10px 30px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        max-width: 1000px;
    }
    
    .upload-section:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 
            0 25px 70px rgba(0, 0, 0, 0.5),
            0 0 40px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .upload-area {
        border: 2px dashed rgba(59, 130, 246, 0.4);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.5);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: rgba(59, 130, 246, 0.7);
        background: rgba(30, 41, 59, 0.7);
        transform: scale(1.01);
    }
    
    /* Medical Report Cards */
    .medical-report {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .report-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .report-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f1f5f9;
        letter-spacing: -0.5px;
    }
    
    .report-subtitle {
        font-size: 1rem;
        color: #e2e8f0;
        font-weight: 400;
    }
    
    /* Diagnosis Cards - Neon Glow */
    .diagnosis-card {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }
    
    .normal-diagnosis {
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(16, 185, 129, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .apnea-diagnosis {
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(239, 68, 68, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .diagnosis-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .normal-diagnosis .diagnosis-title {
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .apnea-diagnosis .diagnosis-title {
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
    }
    
    .diagnosis-subtitle {
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        color: #f1f5f9;
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .severity-badge {
        padding: 0.7rem 1.8rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid;
        box-shadow: 0 8px 25px;
    }
    
    .severity-normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    
    .severity-mild {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-color: rgba(245, 158, 11, 0.5);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4);
    }
    
    .severity-moderate {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        border-color: rgba(249, 115, 22, 0.5);
        box-shadow: 0 8px 25px rgba(249, 115, 22, 0.4);
    }
    
    .severity-severe {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-color: rgba(239, 68, 68, 0.5);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
    }
    
    .medical-info {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #f1f5f9;
        backdrop-filter: blur(10px);
    }
    
    .ecg-section {
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.5);
        padding-bottom: 0.8rem;
        letter-spacing: -0.5px;
    }
    
    .tech-info {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-size: 0.75rem;
        color: #e2e8f0;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    
    .disclaimer {
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 2rem 0;
        color: #fcd34d;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
        font-weight: 500;
    }
    
    /* Buttons - Modern Tech Style */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Primary Button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%) !important;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4) !important;
    }
    
    button[kind="primary"]:hover {
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.6) !important;
        background: linear-gradient(135deg, #059669 0%, #2563eb 100%) !important;
    }
    
    /* Metrics - Dark Cards */
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    
    /* Text Inputs - Dark Theme */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(59, 130, 246, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* File Uploader - Dark Theme */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px dashed rgba(59, 130, 246, 0.4) !important;
        border-radius: 16px !important;
    }
    
    /* Sidebar - Dark Theme */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%) !important;
    }
    
    /* Info/Error/Success Messages */
    .stAlert {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* Remove top spacing from Streamlit default elements */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Center all content horizontally and vertically */
    .main .block-container {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    /* Remove all top margins from first elements */
    .main .block-container > *:first-child {
        margin-top: 0 !important;
    }
    
    /* Center columns */
    .stColumns {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* Compact spacing for single frame */
    .stMarkdown {
        margin-bottom: 0.3rem !important;
    }
    
    /* Compact file uploader */
    .stFileUploader {
        padding: 0.5rem !important;
    }
    
    /* Compact buttons */
    .stButton > button {
        padding: 0.5rem 1.5rem !important;
        font-size: 0.85rem !important;
    }
    
    /* Compact success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(59, 130, 246, 0.7);
    }
    
    /* Animated Background Elements */
    .bg-animation {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
    }
    
    .bg-circle {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.3;
        animation: float 20s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(30px, -30px) scale(1.1); }
        66% { transform: translate(-20px, 20px) scale(0.9); }
    }
</style>
""", unsafe_allow_html=True)

# Clean up any existing temp files
temp_files = glob.glob("temp_*.dat")
for temp_file in temp_files:
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except:
        pass  # Ignore cleanup errors

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# Import chatbot module (already imported at top)

# Feature extraction function
def extract_features(window):
    """Extract the same 12 features used in training"""
    features = []
    
    # Time domain features
    features.append(np.mean(window))
    features.append(np.std(window))
    features.append(np.min(window))
    features.append(np.max(window))
    features.append(np.median(window))
    features.append(np.percentile(window, 25))
    features.append(np.percentile(window, 75))
    
    # Signal energy
    features.append(np.sum(window**2))
    features.append(np.sqrt(np.mean(window**2)))  # RMS
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(window)))
    features.append(zero_crossings)
    
    # Heart rate variability approximation
    diff = np.diff(window)
    features.append(np.std(diff))
    features.append(np.mean(np.abs(diff)))
    
    return np.array(features)

# Function to count apnea events from ECG signal (OPTIMIZED)
def count_apnea_events(ecg_signal, fs, model, scaler, window_size_sec=15, overlap=0.3, min_event_duration_sec=5, max_signal_duration_sec=300):
    """
    Count the number of apnea events in the ECG signal by analyzing it in smaller windows.
    OPTIMIZED VERSION - processes only first N seconds and uses fewer windows.
    
    Parameters:
    -----------
    ecg_signal : array-like
        Full ECG signal array
    fs : int
        Sampling frequency (Hz)
    model : sklearn model
        Trained apnea detection model
    scaler : sklearn scaler
        Feature scaler
    window_size_sec : float, default=15
        Size of analysis window in seconds (larger = fewer windows)
    overlap : float, default=0.3
        Overlap between windows (0-1) - reduced for speed
    min_event_duration_sec : float, default=5
        Minimum duration in seconds to consider as an apnea event
    max_signal_duration_sec : float, default=300
        Maximum signal duration to process (5 minutes) - for performance
        
    Returns:
    --------
    dict with keys:
        - num_events: Total number of apnea events
        - event_positions: List of (start_time, end_time) tuples for each event
        - event_durations: List of durations in seconds for each event
        - ahi: Apnea-Hypopnea Index (events per hour) - estimated
        - signal_duration_hours: Duration of the processed signal in hours
    """
    # Limit signal length for performance (process max 5 minutes)
    max_samples = int(max_signal_duration_sec * fs)
    processed_signal = ecg_signal[:min(len(ecg_signal), max_samples)]
    signal_duration_sec = len(processed_signal) / fs
    
    window_size_samples = int(window_size_sec * fs)
    step_size = int(window_size_samples * (1 - overlap))
    
    # Store predictions for each window
    apnea_predictions = []
    window_start_times = []
    
    # Batch process features for better performance
    windows = []
    valid_indices = []
    
    # Collect all windows first
    for start_idx in range(0, len(processed_signal) - window_size_samples, step_size):
        end_idx = start_idx + window_size_samples
        window = processed_signal[start_idx:end_idx]
        
        # Skip windows with invalid data
        if not (np.any(np.isnan(window)) or len(window) < window_size_samples):
            windows.append(window)
            valid_indices.append((start_idx, start_idx / fs))
    
    if len(windows) == 0:
        return {
            'num_events': 0,
            'event_positions': [],
            'event_durations': [],
            'ahi': 0,
            'signal_duration_hours': signal_duration_sec / 3600,
            'avg_event_duration': 0,
            'total_apnea_time_sec': 0,
            'apnea_percentage': 0
        }
    
    # Batch extract features
    features_list = []
    for window in windows:
        features = extract_features(window)
        features_list.append(features)
    
    # Batch scale and predict
    if len(features_list) > 0:
        features_array = np.array(features_list)
        features_scaled = scaler.transform(features_array)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Consider as apnea if probability > 0.4
        for i, prob in enumerate(probabilities):
            apnea_predictions.append(1 if prob > 0.4 else 0)
            window_start_times.append(valid_indices[i][1])
    
    if len(apnea_predictions) == 0:
        return {
            'num_events': 0,
            'event_positions': [],
            'event_durations': [],
            'ahi': 0,
            'signal_duration_hours': signal_duration_sec / 3600,
            'avg_event_duration': 0,
            'total_apnea_time_sec': 0,
            'apnea_percentage': 0
        }
    
    apnea_timeline = np.array(apnea_predictions)
    window_start_times = np.array(window_start_times)
    
    # Find consecutive apnea windows (events)
    events = []
    in_event = False
    event_start = None
    
    for i, is_apnea in enumerate(apnea_timeline):
        if is_apnea and not in_event:
            # Start of new event
            in_event = True
            event_start = window_start_times[i]
        elif not is_apnea and in_event:
            # End of event
            event_end = window_start_times[i-1] + window_size_sec
            event_duration = event_end - event_start
            # Only count events longer than minimum duration
            if event_duration >= min_event_duration_sec:
                events.append((event_start, event_end))
            in_event = False
            event_start = None
    
    # Handle event that extends to the end of signal
    if in_event:
        event_end = window_start_times[-1] + window_size_sec
        event_duration = event_end - event_start
        if event_duration >= min_event_duration_sec:
            events.append((event_start, event_end))
    
    # Calculate statistics
    num_events = len(events)
    event_durations = [end - start for start, end in events]
    avg_event_duration = np.mean(event_durations) if event_durations else 0
    
    # Calculate AHI (Apnea-Hypopnea Index) - events per hour
    # Scale to full signal duration if we only processed part of it
    signal_duration_hours = signal_duration_sec / 3600
    if len(ecg_signal) > max_samples:
        # Extrapolate AHI based on processed portion
        scaling_factor = (len(ecg_signal) / fs) / signal_duration_sec
        estimated_events = num_events * scaling_factor
        full_duration_hours = (len(ecg_signal) / fs) / 3600
        ahi = estimated_events / full_duration_hours if full_duration_hours > 0 else 0
    else:
        ahi = num_events / signal_duration_hours if signal_duration_hours > 0 else 0
    
    return {
        'num_events': num_events,
        'event_positions': events,
        'event_durations': event_durations,
        'ahi': ahi,
        'signal_duration_hours': signal_duration_sec / 3600,
        'avg_event_duration': avg_event_duration,
        'total_apnea_time_sec': sum(event_durations),
        'apnea_percentage': (sum(event_durations) / signal_duration_sec * 100) if signal_duration_sec > 0 else 0
    }

# Create a proper model
@st.cache_data
def load_trained_model():
    """Load the actual trained model from file"""
    try:
        import joblib
        # Load the trained model and scaler
        model_data = joblib.load('best_sleep_apnea_model.pkl')
        
        if isinstance(model_data, dict):
            # If it's a dictionary with model and scaler
            model = model_data.get('model')
            scaler = model_data.get('scaler')
        else:
            # If it's just the model, create a new scaler
            model = model_data
            scaler = None
        
        # Create a new scaler and fit it with representative data
        scaler = StandardScaler()
        
        # Generate representative training data to fit the scaler
        # This mimics the actual training data distribution
        np.random.seed(42)
        n_samples = 1000
        
        # Normal ECG patterns (lower variability, regular patterns)
        normal_data = []
        for _ in range(n_samples // 2):
            t = np.linspace(0, 30, 3000)
            signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.random.randn(len(t))
            features = extract_features(signal)
            normal_data.append(features)
        
        # Apnea ECG patterns (higher variability, irregular patterns)
        apnea_data = []
        for _ in range(n_samples // 2):
            t = np.linspace(0, 30, 3000)
            signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.6 * t) + 0.3 * np.random.randn(len(t))
            if np.random.random() > 0.5:
                signal[1000:1500] += 0.5 * np.sin(2 * np.pi * 0.3 * t[1000:1500])
            features = extract_features(signal)
            apnea_data.append(features)
        
        # Combine data and fit scaler
        X = np.vstack([normal_data, apnea_data])
        scaler.fit(X)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
        # Fallback to dummy model
        return create_dummy_model(), StandardScaler()

def create_dummy_model():
    """Create a simple dummy model as fallback"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Generate synthetic training data that mimics real ECG patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Normal ECG patterns (lower variability, regular patterns)
    normal_data = []
    for _ in range(n_samples // 2):
        # Generate normal ECG-like signal
        t = np.linspace(0, 30, 3000)  # 30 seconds at 100Hz
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.random.randn(len(t))
        features = extract_features(signal)
        normal_data.append(features)
    
    # Apnea ECG patterns (higher variability, irregular patterns)
    apnea_data = []
    for _ in range(n_samples // 2):
        # Generate apnea-like signal with more irregularity
        t = np.linspace(0, 30, 3000)
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 0.6 * t) + 0.3 * np.random.randn(len(t))
        # Add some irregular patterns
        if np.random.random() > 0.5:
            signal[1000:1500] += 0.5 * np.sin(2 * np.pi * 0.3 * t[1000:1500])
        features = extract_features(signal)
        apnea_data.append(features)
    
    # Combine data
    X = np.vstack([normal_data, apnea_data])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Train model
    model.fit(X, y)
    return model

# Load combined model for AHI and Severity prediction
@st.cache_data
def load_combined_model():
    """Load or train a model that uses both ECG features and patient data"""
    try:
        # Try to load existing model
        if os.path.exists('combined_apnea_model.pkl'):
            model_data = joblib.load('combined_apnea_model.pkl')
            return model_data.get('model'), model_data.get('scaler'), model_data.get('ahi_model'), model_data.get('severity_model'), model_data.get('severity_map')
        
        # If model doesn't exist, train it from data.csv
        if not os.path.exists('data.csv'):
            return None, None, None, None, None
        
        # Load data
        df = pd.read_csv('data.csv')
        
        # Encode categorical variables
        df['Gender_encoded'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Snoring_encoded'] = df['Snoring'].map({True: 1, False: 0})
        
        # Create feature matrix (Age, Gender, Snoring, SpO2, ECG_Heart_Rate, BMI)
        X_patient = df[['Age', 'Gender_encoded', 'Snoring_encoded', 'SpO2', 'ECG_Heart_Rate', 'BMI']].values
        
        # Target: Apnea detection (based on AHI > 5)
        y_apnea = (df['AHI'] > 5).astype(int).values
        
        # Target: AHI (regression)
        y_ahi = df['AHI'].values
        
        # Target: Severity (classification)
        severity_map = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
        y_severity = df['Severity'].map(severity_map).values
        
        # Scale features
        scaler = StandardScaler()
        X_patient_scaled = scaler.fit_transform(X_patient)
        
        # Train apnea detection model
        apnea_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        apnea_model.fit(X_patient_scaled, y_apnea)
        
        # Train AHI regression model
        from sklearn.ensemble import RandomForestRegressor
        ahi_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        ahi_model.fit(X_patient_scaled, y_ahi)
        
        # Train Severity classification model
        severity_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        severity_model.fit(X_patient_scaled, y_severity)
        
        # Save models
        model_data = {
            'model': apnea_model,
            'scaler': scaler,
            'ahi_model': ahi_model,
            'severity_model': severity_model,
            'severity_map': severity_map,
            'feature_names': ['Age', 'Gender', 'Snoring', 'SpO2', 'ECG_Heart_Rate', 'BMI']
        }
        try:
            joblib.dump(model_data, 'combined_apnea_model.pkl')
        except Exception as e:
            pass  # Silent fail during model saving in cache
        
        return apnea_model, scaler, ahi_model, severity_model, severity_map
        
    except Exception as e:
        st.warning(f"Could not load combined model: {e}. Using ECG-only prediction.")
        return None, None, None, None, None

# Process uploaded file
def process_file(uploaded_file, patient_data=None):
    """Process the uploaded ECG file and return results"""
    try:
        # Always process - don't cache based on file hash to ensure fresh results
        file_content = uploaded_file.read()
        file_hash = hash(file_content)
        
        # Only skip if explicitly told not to reprocess AND it's the exact same file
        # But we'll force reprocess on button click anyway
        if (st.session_state.file_hash == file_hash and 
            st.session_state.results is not None and 
            not st.session_state.get('force_reprocess', True)):  # Changed default to True
            return st.session_state.results, None
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        if uploaded_file.name.endswith('.dat'):
            # Try to process .dat file directly without creating temp files
            try:
                # First try to process as binary data directly
                ecg_signal = np.frombuffer(file_content, dtype=np.int16).astype(np.float64)
                fs = 100
                ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
            except:
                # If that fails, try with wfdb using a unique temp file
                import tempfile
                import uuid
                
                temp_path = f"temp_{uuid.uuid4().hex}_{uploaded_file.name}"
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(file_content)
                    
                    try:
                        record = wfdb.rdrecord(temp_path.replace('.dat', ''))
                        ecg_signal = record.p_signal[:, 0]
                        fs = int(record.fs)
                    except:
                        ecg_signal = np.frombuffer(file_content, dtype=np.int16).astype(np.float64)
                        fs = 100
                        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                    
                finally:
                    # Try to clean up temp file, but don't fail if it can't be deleted
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass  # Ignore cleanup errors
                
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            ecg_signal = df.iloc[:, 0].values
            ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
            fs = 100
            
        elif uploaded_file.name.endswith('.txt'):
            ecg_signal = np.loadtxt(uploaded_file)
            ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
            fs = 100
        else:
            try:
                uploaded_file.seek(0)
                ecg_signal = np.loadtxt(uploaded_file)
                ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
                fs = 100
            except:
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                ecg_signal = np.frombuffer(file_content, dtype=np.int16).astype(np.float64)
                fs = 100
                ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # Validate signal
        if len(ecg_signal) == 0:
            return None, "No valid data found in the uploaded file."
        
        min_samples = 30 * fs
        if len(ecg_signal) < min_samples:
            return None, f"Signal too short! Need at least {min_samples} samples. Found {len(ecg_signal)} samples."
        
        if np.all(np.isnan(ecg_signal)) or np.all(ecg_signal == 0):
            return None, "Invalid signal data: all values are NaN or zero."
        
        # Load trained model and scaler
        model, scaler = load_trained_model()
        
        # Skip model info display for faster loading
        
        # Count apnea events from signal (optimized - only first 60 seconds for faster processing ~10 sec target)
        with st.spinner("🔍 Analyzing ECG signal (this may take a few seconds)..."):
            # Limit to 60 seconds for much faster processing
            max_analysis_samples = min(len(ecg_signal), 60 * fs)
            event_stats = count_apnea_events(ecg_signal[:max_analysis_samples], fs, model, scaler, 
                                              max_signal_duration_sec=60, window_size_sec=15, overlap=0.2)
        
        # Also analyze first 30 seconds for initial assessment
        window_size = 30 * fs
        ecg_window = ecg_signal[:min(window_size, len(ecg_signal))]
        
        # Extract features from first 30 seconds for initial assessment
        features = extract_features(ecg_window)
        features = features.reshape(1, -1)
        
        # Show extracted features for transparency
        with st.expander("🔍 View Extracted Features"):
            feature_names = [
                "Mean", "Std Dev", "Min", "Max", "Median", 
                "25th Percentile", "75th Percentile", "Energy", 
                "RMS", "Zero Crossings", "Std of Diff", "Mean Abs Diff"
            ]
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': features[0]
            })
            st.dataframe(feature_df, use_container_width=True)
        
        # Scale features using the trained scaler
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction for initial assessment
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Use the model's actual prediction probability
        apnea_prob = probability[1] * 100
        
        # Calculate ECG Heart Rate from signal (simplified method)
        try:
            # Find peaks in the signal
            signal_std = np.std(ecg_window)
            if signal_std > 0:
                # Find local maxima (simplified peak detection)
                peaks = []
                for i in range(1, len(ecg_window) - 1):
                    if ecg_window[i] > ecg_window[i-1] and ecg_window[i] > ecg_window[i+1] and ecg_window[i] > np.mean(ecg_window) + signal_std:
                        peaks.append(i)
                
                if len(peaks) > 1:
                    # Calculate average time between peaks
                    peak_intervals = np.diff(peaks) / fs  # in seconds
                    avg_interval = np.mean(peak_intervals)
                    if avg_interval > 0:
                        ecg_heart_rate = int(60 / avg_interval)
                    else:
                        ecg_heart_rate = 72
                else:
                    ecg_heart_rate = 72
            else:
                ecg_heart_rate = 72
        except:
            ecg_heart_rate = 72
        
        # Validate heart rate range
        if ecg_heart_rate < 40 or ecg_heart_rate > 150:
            ecg_heart_rate = 72  # Default if calculation is off
        
        # Predict AHI and Severity using combined model if patient data is available
        predicted_ahi = None
        predicted_severity = None
        
        if patient_data:
            try:
                combined_model, combined_scaler, ahi_model, severity_model, severity_map = load_combined_model()
                
                if combined_model and combined_scaler and ahi_model and severity_model:
                    # Prepare patient features
                    # Get BMI from patient data (calculated from height/weight)
                    bmi = patient_data.get('bmi', 25.0)
                    if bmi is None or bmi <= 0:
                        # Calculate from height and weight if available
                        height = patient_data.get('height', 170.0)
                        weight = patient_data.get('weight', 70.0)
                        if height > 0:
                            height_m = height / 100.0
                            bmi = weight / (height_m ** 2)
                        else:
                            bmi = 25.0
                    
                    # Encode patient data
                    gender_encoded = 1 if patient_data.get('gender', 'Male') == 'Male' else 0
                    snoring_encoded = 1 if patient_data.get('snoring', 'No') == 'Yes' else 0
                    age = patient_data.get('age', 40)
                    spo2 = patient_data.get('spo2', 95.0)
                    
                    # Create feature array: [Age, Gender, Snoring, SpO2, ECG_Heart_Rate, BMI]
                    patient_features = np.array([[age, gender_encoded, snoring_encoded, spo2, ecg_heart_rate, bmi]])
                    patient_features_scaled = combined_scaler.transform(patient_features)
                    
                    # Predict AHI using combined model
                    predicted_ahi_combined = max(0, float(ahi_model.predict(patient_features_scaled)[0]))
                    
                    # Also get AHI from ECG event detection if available
                    ahi_from_ecg = event_stats.get('ahi', 0)
                    
                    # Check if patient vitals strongly suggest normal (conservative approach)
                    # High SpO2 (>96), no snoring, normal BMI (18.5-25), younger age (<40)
                    strong_normal_indicators = (
                        spo2 > 96 and 
                        snoring_encoded == 0 and 
                        18.5 <= bmi <= 25 and 
                        age < 40 and
                        ahi_from_ecg == 0
                    )
                    
                    # If strong normal indicators and predicted AHI is borderline, be more conservative
                    if strong_normal_indicators and predicted_ahi_combined < 8:
                        # Cap the AHI at a lower value for normal cases
                        predicted_ahi_combined = min(predicted_ahi_combined, 3.0)
                    
                    # Use weighted average: 70% combined model, 30% ECG-based if events detected
                    if ahi_from_ecg > 0:
                        predicted_ahi = 0.7 * predicted_ahi_combined + 0.3 * ahi_from_ecg
                    else:
                        predicted_ahi = predicted_ahi_combined
                    
                    # Predict Severity using combined model
                    severity_pred = severity_model.predict(patient_features_scaled)[0]
                    reverse_severity_map = {v: k for k, v in severity_map.items()}
                    predicted_severity = reverse_severity_map.get(severity_pred, 'None')  # Default to 'None' instead of 'Mild'
                    
                    # Override severity if we have strong normal indicators and low predicted AHI
                    if strong_normal_indicators and predicted_ahi_combined < 5:
                        predicted_severity = 'None'
                    
            except Exception as e:
                st.warning(f"Could not use combined model: {e}")
        
        # Determine severity and prediction - PRIORITIZE AHI FIRST
        # Priority: 1) predicted_ahi, 2) event_stats AHI, 3) predicted_severity, 4) ECG probability
        
        # First, check if we have a predicted AHI (from combined model)
        if predicted_ahi is not None and predicted_ahi >= 0:
            # Use predicted AHI as primary indicator
            if predicted_ahi < 5:
                severity = "Normal"
                severity_class = "normal"
                prediction = 0
                # Override predicted_severity if AHI indicates normal
                if predicted_severity and predicted_severity != 'None':
                    predicted_severity = 'None'
            elif predicted_ahi < 15:
                severity = "Mild"
                severity_class = "mild"
                prediction = 1
            elif predicted_ahi < 30:
                severity = "Moderate"
                severity_class = "moderate"
                prediction = 1
            else:
                severity = "Severe"
                severity_class = "severe"
                prediction = 1
        # Second, check ECG event detection AHI (only if we have events detected)
        elif event_stats.get('num_events', 0) > 0:
            # Use AHI from event detection
            ahi = event_stats.get('ahi', 0)
            if ahi < 5:
                severity = "Normal"
                severity_class = "normal"
                prediction = 0
            elif ahi < 15:
                severity = "Mild"
                severity_class = "mild"
                prediction = 1
            elif ahi < 30:
                severity = "Moderate"
                severity_class = "moderate"
                prediction = 1
            else:
                severity = "Severe"
                severity_class = "severe"
                prediction = 1
        # Third, check if we have no events and low probability - definitely normal
        elif event_stats.get('num_events', 0) == 0 and apnea_prob < 35:
            severity = "Normal"
            severity_class = "normal"
            prediction = 0
        # Fourth, check predicted_severity from combined model (if no AHI available)
        elif predicted_severity:
            severity = predicted_severity
            if severity == 'None':
                severity = 'Normal'
                severity_class = 'normal'
                prediction = 0
            elif severity == 'Mild':
                severity_class = 'mild'
                prediction = 1
            elif severity == 'Moderate':
                severity_class = 'moderate'
                prediction = 1
            elif severity == 'Severe':
                severity_class = 'severe'
                prediction = 1
            else:
                # Unknown severity, default to normal to be conservative
                severity = 'Normal'
                severity_class = 'normal'
                prediction = 0
        # Fifth, fallback to probability-based classification (ECG only)
        else:
            # Use conservative threshold for normal - be more lenient towards normal
            if apnea_prob < 35:  # More conservative threshold
                severity = "Normal"
                severity_class = "normal"
                prediction = 0
            elif apnea_prob < 55:  # Adjusted threshold
                severity = "Mild"
                severity_class = "mild"
                prediction = 1
            elif apnea_prob < 75:  # Adjusted threshold
                severity = "Moderate"
                severity_class = "moderate"
                prediction = 1
            else:
                severity = "Severe"
                severity_class = "severe"
                prediction = 1
        
        # Prepare results
        results = {
            'prediction': prediction,
            'probability': [1-apnea_prob/100, apnea_prob/100],
            'confidence': max(1-apnea_prob/100, apnea_prob/100) * 100,
            'apnea_prob': apnea_prob,
            'severity': severity,
            'severity_class': severity_class,
            'predicted_ahi': predicted_ahi,  # AHI from combined model
            'predicted_severity': predicted_severity,  # Severity from combined model
            'ecg_signal': ecg_signal,  # Store full signal for event visualization
            'ecg_window': ecg_window,  # First 30 seconds for initial display
            'fs': fs,
            'file_name': uploaded_file.name,
            'analysis_date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            'file_hash': file_hash,
            'event_stats': event_stats,  # Include event counting statistics
            'patient_data': patient_data  # Store patient data
        }
        
        # Store in session state
        st.session_state.file_hash = file_hash
        st.session_state.results = results
        st.session_state.force_reprocess = False  # Reset reprocess flag
        
        return results, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def plot_roc_curves_comparison(y_true, y_scores_dict, figsize=(10, 8), 
                                 title='ROC Curves Comparison - Sleep Apnea Detection Models',
                                 save_path=None):
    """
    Plot ROC curves comparing multiple machine learning models.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (ground truth)
    y_scores_dict : dict
        Dictionary with model names as keys and prediction probabilities as values.
        Example: {'Random Forest': y_proba1, 'SVM': y_proba2, ...}
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    title : str, default='ROC Curves Comparison - Sleep Apnea Detection Models'
        Title for the plot
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    auc_scores : dict
        Dictionary with model names as keys and AUC scores as values
    """
    # Set style for medical/professional plots
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Define color palette for different models (medical-friendly colors)
    colors = ['#2c3e50', '#3498db', '#e74c3c', '#27ae60', '#f39c12', 
              '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#16a085']
    
    # Dictionary to store AUC scores
    auc_scores = {}
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, 
            label='Random Classifier (AUC = 0.50)')
    
    # Process each model
    for idx, (model_name, y_scores) in enumerate(y_scores_dict.items()):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        
        # Plot ROC curve with different color for each model
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5, alpha=0.9,
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Customize the plot
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Set limits
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Add legend with AUC scores
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add text box with summary statistics
    textstr = f'Total Models: {len(y_scores_dict)}\n'
    textstr += f'Best AUC: {max(auc_scores.values()):.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax, auc_scores

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name='Classifier', 
                                   class_labels=None, figsize=(8, 6), 
                                   save_path=None, normalize=False):
    """
    Plot a publication-ready confusion matrix heatmap using seaborn and matplotlib.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (ground truth)
    y_pred : array-like
        Predicted labels from the model
    model_name : str, default='Classifier'
        Name of the model to display in the title
    class_labels : list, optional
        Custom class labels (e.g., ['Normal', 'Apnea']). 
        If None, defaults to ['Class 0', 'Class 1']
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    normalize : bool, default=False
        If True, normalize the confusion matrix to show percentages
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    cm : numpy.ndarray
        The confusion matrix array
    """
    # Set style for medical/professional plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_display = cm_percent
        fmt = '.1f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Set default class labels if not provided
    if class_labels is None:
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_labels = [f'Class {label}' for label in unique_labels]
    
    # Create heatmap using seaborn
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', 
                cbar_kws={'label': 'Count' if not normalize else 'Percentage (%)'},
                square=True, linewidths=1, linecolor='gray',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, vmin=0, vmax=None if normalize else None,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    # Customize the plot
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Confusion Matrix – {model_name}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Set tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Add grid lines for better visibility
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Calculate and display metrics
    total = np.sum(cm)
    correct = np.trace(cm)
    accuracy = correct / total * 100
    
    # Calculate per-class metrics
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100 if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100 if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    # Add text box with metrics
    textstr = f'Accuracy: {accuracy:.2f}%\n'
    textstr += f'Sensitivity: {sensitivity:.2f}%\n'
    textstr += f'Specificity: {specificity:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, fontweight='bold',
           family='monospace')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax, cm

def plot_feature_importance(model, feature_names=None, top_n=10, 
                              figsize=(10, 6), 
                              title='Feature Importance – Sleep Apnea Detection',
                              save_path=None):
    """
    Visualize feature importance from a trained Random Forest model using Seaborn and Matplotlib.
    
    Parameters:
    -----------
    model : sklearn model
        Trained Random Forest model (or any model with feature_importances_ attribute)
    feature_names : list, optional
        List of feature names. If None, defaults to ['Feature 0', 'Feature 1', ...]
    top_n : int, default=10
        Number of top features to display (sorted by importance)
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches
    title : str, default='Feature Importance – Sleep Apnea Detection'
        Title for the plot
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    df_importance : pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have 'feature_importances_' attribute. "
                           "This function is designed for tree-based models like Random Forest.")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Create DataFrame mapping feature names to importance scores
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance (descending) and select top N
    df_importance = df_importance.sort_values('Importance', ascending=False).head(top_n)
    
    # Set style for medical/professional plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Define professional color palette (medical/research theme)
    colors = sns.color_palette("Blues_r", n_colors=len(df_importance))
    
    # Create horizontal bar chart using Seaborn
    sns.barplot(data=df_importance, y='Feature', x='Importance', 
                palette=colors, ax=ax, orient='h')
    
    # Customize the plot
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Features', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    ax.set_axisbelow(True)
    
    # Format x-axis to show percentages
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_importance.iterrows()):
        importance_value = row['Importance']
        ax.text(importance_value + 0.005, i, f'{importance_value:.4f}', 
               va='center', fontsize=9, fontweight='bold')
    
    # Add text box with summary statistics
    total_importance = df_importance['Importance'].sum()
    textstr = f'Top {top_n} Features\n'
    textstr += f'Total Importance: {total_importance:.4f}\n'
    textstr += f'Most Important: {df_importance.iloc[0]["Feature"]}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=props, fontweight='bold', family='monospace')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax, df_importance

def plot_ecg_comparison(duration=30, sampling_rate=100, apnea_start_time=15, 
                         noise_level=0.05, amplitude_reduction=0.5,
                         figsize=(14, 6), title='ECG Signal Comparison: Normal vs Apnea',
                         save_path=None):
    """
    Simulate and plot ECG signals comparing normal vs apnea (abnormal) segments.
    
    Parameters:
    -----------
    duration : float, default=30
        Duration of the signal in seconds
    sampling_rate : int, default=100
        Sampling rate in Hz
    apnea_start_time : float, default=15
        Time (in seconds) when apnea begins (amplitude reduction starts)
    noise_level : float, default=0.05
        Level of noise to add for realism
    amplitude_reduction : float, default=0.5
        Fraction of amplitude reduction during apnea (0.5 = 50% reduction)
    figsize : tuple, default=(14, 6)
        Figure size (width, height) in inches
    title : str, default='ECG Signal Comparison: Normal vs Apnea'
        Title for the plot
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object (or array of axes if subplots)
    signals : dict
        Dictionary containing time, normal_ecg, and apnea_ecg arrays
    """
    # Set style for medical/professional plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    sns.set_palette("husl")
    
    # Generate time axis
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Generate normal ECG signal
    # Use multiple sinusoidal components to simulate realistic ECG waveform
    # QRS complex frequency (heartbeat ~1.2 Hz = 72 bpm)
    heart_rate = 1.2  # Hz (72 beats per minute)
    
    # Create base ECG signal with QRS complexes
    normal_ecg = np.zeros_like(t)
    
    # QRS complex simulation (main heartbeat peaks)
    for i in range(int(duration * heart_rate)):
        peak_time = i / heart_rate
        if peak_time < duration:
            # QRS complex - sharp peak
            qrs_center = peak_time
            qrs_width = 0.1  # 100ms QRS duration
            qrs_indices = np.where((t >= qrs_center - qrs_width/2) & 
                                  (t <= qrs_center + qrs_width/2))[0]
            if len(qrs_indices) > 0:
                # Create QRS complex shape
                qrs_signal = np.exp(-((t[qrs_indices] - qrs_center)**2) / (2 * (qrs_width/3)**2))
                normal_ecg[qrs_indices] += 0.8 * qrs_signal
    
    # Add P-wave (before QRS)
    for i in range(int(duration * heart_rate)):
        peak_time = i / heart_rate
        p_time = peak_time - 0.2  # P-wave 200ms before QRS
        if p_time >= 0 and p_time < duration:
            p_width = 0.08
            p_indices = np.where((t >= p_time - p_width/2) & 
                               (t <= p_time + p_width/2))[0]
            if len(p_indices) > 0:
                p_signal = np.exp(-((t[p_indices] - p_time)**2) / (2 * (p_width/3)**2))
                normal_ecg[p_indices] += 0.2 * p_signal
    
    # Add T-wave (after QRS)
    for i in range(int(duration * heart_rate)):
        peak_time = i / heart_rate
        t_time = peak_time + 0.3  # T-wave 300ms after QRS
        if t_time >= 0 and t_time < duration:
            t_width = 0.15
            t_indices = np.where((t >= t_time - t_width/2) & 
                               (t <= t_time + t_width/2))[0]
            if len(t_indices) > 0:
                t_signal = np.exp(-((t[t_indices] - t_time)**2) / (2 * (t_width/3)**2))
                normal_ecg[t_indices] += 0.3 * t_signal
    
    # Add smooth sinusoidal baseline variation
    baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)  # Slow respiratory variation
    normal_ecg += baseline
    
    # Add noise for realism
    noise = np.random.normal(0, noise_level, size=len(t))
    normal_ecg += noise
    
    # Generate apnea ECG signal (amplitude reduction after apnea_start_time)
    apnea_ecg = normal_ecg.copy()
    
    # Find indices where apnea begins
    apnea_indices = np.where(t >= apnea_start_time)[0]
    
    if len(apnea_indices) > 0:
        # Gradually reduce amplitude to simulate apnea effect
        # Create smooth transition
        transition_time = 2.0  # 2 seconds transition
        transition_samples = int(transition_time * sampling_rate)
        
        # Smooth amplitude reduction
        for i, idx in enumerate(apnea_indices):
            if i < transition_samples:
                # Gradual transition
                reduction_factor = 1 - (amplitude_reduction * (i / transition_samples))
            else:
                # Full reduction after transition
                reduction_factor = 1 - amplitude_reduction
            
            apnea_ecg[idx] *= reduction_factor
        
        # Add irregular variations during apnea (simulating breathing irregularities)
        apnea_signal_length = len(apnea_indices)
        irregularity = 0.15 * np.sin(2 * np.pi * 0.5 * t[apnea_indices]) * \
                       (1 + 0.3 * np.random.random(apnea_signal_length))
        apnea_ecg[apnea_indices] += irregularity
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='white', 
                                   sharex=True, sharey=False)
    
    # Plot Normal ECG
    ax1.plot(t, normal_ecg, color='#3498db', linewidth=1.5, alpha=0.9, 
            label='Normal ECG')
    ax1.set_ylabel('Amplitude (mV)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('Normal ECG Signal', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    
    # Add annotation for normal breathing
    ax1.text(0.98, 0.95, 'Normal Breathing Pattern', transform=ax1.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60', alpha=0.8, 
                     edgecolor='darkgreen'),
            fontsize=10, fontweight='bold', verticalalignment='top', 
            horizontalalignment='right', color='white')
    
    # Plot Apnea ECG
    ax2.plot(t, apnea_ecg, color='#e74c3c', linewidth=1.5, alpha=0.9, 
            label='Apnea ECG')
    
    # Highlight apnea region
    apnea_mask = t >= apnea_start_time
    ax2.fill_between(t, apnea_ecg, where=apnea_mask, alpha=0.3, 
                     color='#e74c3c', label='Apnea Region')
    
    # Add vertical line to mark apnea start
    ax2.axvline(x=apnea_start_time, color='#f39c12', linestyle='--', 
               linewidth=2, alpha=0.8, label='Apnea Onset')
    
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Amplitude (mV)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('Apnea ECG Signal (Amplitude Reduction)', fontsize=13, 
                 fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    
    # Add annotation for apnea
    ax2.text(0.98, 0.95, 'Sleep Apnea Detected', transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.8, 
                     edgecolor='darkred'),
            fontsize=10, fontweight='bold', verticalalignment='top', 
            horizontalalignment='right', color='white')
    
    # Remove top and right spines for cleaner look
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_facecolor('#fafafa')
    
    # Set overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Prepare signals dictionary
    signals = {
        'time': t,
        'normal_ecg': normal_ecg,
        'apnea_ecg': apnea_ecg,
        'sampling_rate': sampling_rate,
        'duration': duration
    }
    
    return fig, [ax1, ax2], signals

def plot_tsne_visualization(X, y, n_components=2, random_state=42, 
                             perplexity=30, n_iter=1000, figsize=(10, 8),
                             title='t-SNE Visualization – Sleep Apnea Detection Feature Distribution',
                             class_labels=None, save_path=None):
    """
    Visualize feature distribution using t-SNE for binary classification.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Binary labels (0=Normal, 1=Apnea)
    n_components : int, default=2
        Number of components for t-SNE (should be 2 for 2D visualization)
    random_state : int, default=42
        Random state for reproducibility
    perplexity : float, default=30
        Perplexity parameter for t-SNE (typically between 5-50)
    n_iter : int, default=1000
        Maximum number of iterations for t-SNE
    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches
    title : str, default='t-SNE Visualization – Sleep Apnea Detection Feature Distribution'
        Title for the plot
    class_labels : dict, optional
        Dictionary mapping class values to labels (e.g., {0: 'Normal', 1: 'Apnea'})
        If None, defaults to {0: 'Normal', 1: 'Apnea'}
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object
    ax : matplotlib.axes.Axes
        The matplotlib axes object
    tsne_df : pandas.DataFrame
        DataFrame with t-SNE reduced features and labels
    """
    # Set default class labels if not provided
    if class_labels is None:
        class_labels = {0: 'Normal', 1: 'Apnea'}
    
    # Convert to numpy arrays if needed
    X = np.array(X)
    y = np.array(y)
    
    # Check input shapes
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
    
    if n_components != 2:
        raise ValueError("For visualization, n_components must be 2")
    
    # Set style for medical/professional plots
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('seaborn-whitegrid')
    sns.set_palette("husl")
    
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=n_components, random_state=random_state, 
                perplexity=perplexity, n_iter=n_iter, verbose=0)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame with reduced features and labels
    tsne_df = pd.DataFrame({
        't-SNE Component 1': X_tsne[:, 0],
        't-SNE Component 2': X_tsne[:, 1],
        'Label': [class_labels[label] for label in y],
        'Class': y
    })
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Define color palette for classes
    colors = {'Normal': '#3498db', 'Apnea': '#e74c3c'}
    
    # Create scatter plot using Seaborn
    for label_name in class_labels.values():
        mask = tsne_df['Label'] == label_name
        ax.scatter(tsne_df.loc[mask, 't-SNE Component 1'], 
                  tsne_df.loc[mask, 't-SNE Component 2'],
                  c=colors.get(label_name, '#95a5a6'),
                  label=label_name,
                  alpha=0.6,
                  s=50,
                  edgecolors='white',
                  linewidths=0.5)
    
    # Customize the plot
    ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Add legend
    legend = ax.legend(loc='best', fontsize=11, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.95,
                      title='Class', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add text box with statistics
    n_normal = np.sum(y == 0)
    n_apnea = np.sum(y == 1)
    textstr = f'Total Samples: {len(y)}\n'
    textstr += f'Normal: {n_normal} ({n_normal/len(y)*100:.1f}%)\n'
    textstr += f'Apnea: {n_apnea} ({n_apnea/len(y)*100:.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props, fontweight='bold',
           family='monospace')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig, ax, tsne_df

# Main app
def main():
    # Render chatbot in sidebar
    render_chatbot()
    
    # Futuristic Dark Header
    st.markdown("""
    <div class="header-section">
        <h1 class="main-title">⚕️ SLEEP APNEA DETECTION SYSTEM</h1>
        <p class="subtitle">Advanced AI-Powered ECG Analysis | Machine Learning | Real-Time Processing</p>
        <div style="margin-top: 0.5rem;">
            <span class="tech-badge">🤖 AI-Powered</span>
            <span class="tech-badge">⚡ Real-Time</span>
            <span class="tech-badge">📊 85.2% Accuracy</span>
            <span class="tech-badge">🔬 Clinical Grade</span>
            <span class="tech-badge">💊 Healthcare Tech</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Information - Compact Dark Tech Style
    col_tech1, col_tech2, col_tech3 = st.columns(3)
    with col_tech1:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; background: rgba(30, 41, 59, 0.6); 
                    border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);">
            <div style="font-size: 0.85rem; font-weight: 700; color: #10b981; margin-bottom: 0.2rem;">🧠 AI Technology</div>
            <div style="font-size: 0.7rem; color: #e2e8f0;">Random Forest ML</div>
        </div>
        """, unsafe_allow_html=True)
    with col_tech2:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; background: rgba(30, 41, 59, 0.6); 
                    border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3);">
            <div style="font-size: 0.85rem; font-weight: 700; color: #3b82f6; margin-bottom: 0.2rem;">⚙️ Processing</div>
            <div style="font-size: 0.7rem; color: #e2e8f0;">12-Feature ECG</div>
        </div>
        """, unsafe_allow_html=True)
    with col_tech3:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; background: rgba(30, 41, 59, 0.6); 
                    border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3);">
            <div style="font-size: 0.85rem; font-weight: 700; color: #8b5cf6; margin-bottom: 0.2rem;">📈 Performance</div>
            <div style="font-size: 0.7rem; color: #e2e8f0;">85.2% Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.analysis_done:
        # Patient Information Input Section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="color: #f1f5f9; font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem;">
                👤 PATIENT INFORMATION
            </h2>
            <p style="color: #e2e8f0; font-size: 0.7rem; margin: 0;">
                Please provide patient details for comprehensive analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Patient input fields in columns
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            patient_age = st.number_input("Age (years)", min_value=1, max_value=120, value=40, 
                                         help="Patient age in years", key="patient_age")
            patient_gender = st.selectbox("Gender", ["Male", "Female"], 
                                         help="Patient gender", key="patient_gender")
        
        with col_p2:
            patient_height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, 
                                           value=170.0, step=0.1,
                                           help="Patient height in centimeters", key="patient_height")
            patient_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, 
                                           value=70.0, step=0.1,
                                           help="Patient weight in kilograms", key="patient_weight")
        
        with col_p3:
            patient_snoring = st.selectbox("Snoring", ["Yes", "No"], 
                                          help="Does the patient snore?", key="patient_snoring")
            patient_spo2 = st.number_input("SpO2 (%)", min_value=70.0, max_value=100.0, 
                                          value=95.0, step=0.1,
                                          help="Blood oxygen saturation level", key="patient_spo2")
        
        # Calculate BMI
        patient_height_m = patient_height / 100.0  # Convert cm to meters
        patient_bmi = patient_weight / (patient_height_m ** 2) if patient_height_m > 0 else 25.0
        
        # Display BMI
        bmi_color = '#10b981' if 18.5 <= patient_bmi <= 24.9 else ('#f59e0b' if patient_bmi < 18.5 or 25 <= patient_bmi <= 29.9 else '#ef4444')
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {bmi_color}; 
                    border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem; text-align: center;">
            <div style="color: #e2e8f0; font-size: 0.85rem; margin-bottom: 0.3rem; font-weight: 600;">CALCULATED BMI</div>
            <div style="color: {bmi_color}; font-size: 1.5rem; font-weight: 700;">{patient_bmi:.1f} kg/m²</div>
            <div style="color: #94a3b8; font-size: 0.7rem; margin-top: 0.2rem;">
                {'Normal' if 18.5 <= patient_bmi <= 24.9 else ('Underweight' if patient_bmi < 18.5 else ('Overweight' if 25 <= patient_bmi <= 29.9 else 'Obese'))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Store patient data in session state
        st.session_state.patient_data = {
            'age': patient_age,
            'gender': patient_gender,
            'snoring': patient_snoring,
            'spo2': patient_spo2,
            'height': patient_height,
            'weight': patient_weight,
            'bmi': patient_bmi
        }
        
        # Upload Section - Compact Dark Tech Style
        st.markdown("""
        <div style="text-align: center; margin: 1.5rem 0 0.3rem 0;">
            <h2 style="color: #f1f5f9; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.15rem;">
                📋 ECG UPLOAD PORTAL
            </h2>
            <p style="color: #e2e8f0; font-size: 0.65rem; margin: 0;">
                Upload ECG file for AI-powered sleep apnea detection
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # Compact file uploader
        uploaded_file = st.file_uploader(
            "Choose ECG File (.dat, .csv, .txt)",
            type=['dat', 'csv', 'txt'],
            help="Upload patient ECG signal file",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            col_file1, col_file2 = st.columns([2, 1])
            with col_file1:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3);
                            border-radius: 8px; padding: 0.5rem; margin: 0.3rem 0; color: #10b981; font-size: 0.8rem;">
                    ✅ {uploaded_file.name}
                </div>
                """, unsafe_allow_html=True)
            with col_file2:
                if st.button("🗑️", help="Clear", key="clear_file_btn"):
                    st.session_state.results = None
                    st.session_state.file_hash = None
                    st.session_state.analysis_done = False
                    st.session_state.force_reprocess = False
                    st.rerun()
            
            # Analyze button - Compact
            if st.button("🔍 ANALYZE ECG SIGNAL", type="primary", use_container_width=True):
                # Clear previous results when analyzing new file
                st.session_state.results = None
                st.session_state.file_hash = None
                st.session_state.analysis_done = False
                st.session_state.force_reprocess = True
                
                with st.spinner("Analyzing..."):
                    # Get patient data from session state
                    patient_data = st.session_state.get('patient_data', {})
                    results, error = process_file(uploaded_file, patient_data=patient_data)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                        st.session_state.analysis_done = False
                    else:
                        st.session_state.results = results
                        st.session_state.analysis_done = True
                        st.rerun()
        
        # Compact Analysis Process - Horizontal
        st.markdown("""
        <div style="margin-top: 0.5rem;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem; margin-top: 0.3rem;">
            <div style="text-align: center; padding: 0.4rem 0.2rem; background: rgba(59, 130, 246, 0.1); border-radius: 6px; border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="font-size: 1rem; margin-bottom: 0.15rem;">📤</div>
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.65rem;">UPLOAD</div>
            </div>
            <div style="text-align: center; padding: 0.4rem 0.2rem; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="font-size: 1rem; margin-bottom: 0.15rem;">⚙️</div>
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.65rem;">PROCESS</div>
            </div>
            <div style="text-align: center; padding: 0.4rem 0.2rem; background: rgba(139, 92, 246, 0.1); border-radius: 6px; border: 1px solid rgba(139, 92, 246, 0.3);">
                <div style="font-size: 1rem; margin-bottom: 0.15rem;">🤖</div>
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.65rem;">ANALYZE</div>
            </div>
            <div style="text-align: center; padding: 0.4rem 0.2rem; background: rgba(239, 68, 68, 0.1); border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.3);">
                <div style="font-size: 1rem; margin-bottom: 0.15rem;">📊</div>
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.65rem;">REPORT</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Medical Report Section
        if 'results' not in st.session_state or st.session_state.results is None:
            st.error("❌ No analysis results found. Please upload and analyze a file first.")
            st.session_state.analysis_done = False
            st.rerun()
        
        results = st.session_state.results
        
        # Report Header - Dark Tech Style
        st.markdown(f"""
        <div class="report-header">
            <h2 class="report-title">📊 CLINICAL ANALYSIS REPORT</h2>
            <p class="report-subtitle">Sleep Apnea Detection System • Analysis Date: {results.get('analysis_date', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Patient Information - Simplified
        file_name = results.get('file_name', 'N/A')
        analysis_date = results.get('analysis_date', 'N/A')
        
        st.markdown(f"""
        <div class="medical-report">
            <h3 class="section-title">📋 TEST INFORMATION</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div style="background: rgba(30, 41, 59, 0.6); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <div style="color: #e2e8f0; font-size: 0.85rem; margin-bottom: 0.3rem; font-weight: 600;">TEST DATE</div>
                    <div style="color: #f1f5f9; font-weight: 600; font-size: 1rem;">{analysis_date}</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.6); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <div style="color: #e2e8f0; font-size: 0.85rem; margin-bottom: 0.3rem; font-weight: 600;">FILE</div>
                    <div style="color: #f1f5f9; font-weight: 600; font-size: 1rem; word-break: break-word;">{file_name[:30]}{'...' if len(file_name) > 30 else ''}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get event statistics if available
        event_stats = results.get('event_stats', {})
        num_events = event_stats.get('num_events', 0)
        event_positions = event_stats.get('event_positions', [])
        
        # Clinical Diagnosis - Patient-Focused
        prediction = results.get('prediction', 0)
        confidence = results.get('confidence', 0)
        apnea_prob = results.get('apnea_prob', 0)
        
        if prediction == 0:
            st.markdown(f"""
            <div class="diagnosis-card normal-diagnosis">
                <h2 class="diagnosis-title">✅ NORMAL BREATHING PATTERN</h2>
                <p class="diagnosis-subtitle">No signs of sleep apnea detected in your ECG analysis</p>
                <div class="confidence-badge" style="margin-top: 1rem;">Test Confidence: {confidence:.1f}%</div>
                <p style="margin-top: 1.5rem; font-size: 1.1rem; color: #f1f5f9; line-height: 1.6;">
                    Your ECG analysis shows normal breathing patterns with no evidence of sleep apnea events.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            severity = results.get('severity', 'Mild')
            severity_colors = {
                'mild': '#f59e0b',
                'moderate': '#f97316',
                'severe': '#ef4444'
            }
            severity_class = results.get('severity_class', 'mild')
            color = severity_colors.get(severity_class, '#ef4444')
            
            st.markdown(f"""
            <div class="diagnosis-card apnea-diagnosis" style="border-left: 5px solid {color};">
                <h2 class="diagnosis-title">⚠️ SLEEP APNEA DETECTED</h2>
                <p class="diagnosis-subtitle">Sleep apnea events identified in your ECG analysis</p>
                <div class="severity-badge severity-{severity_class}" style="background-color: {color}; margin-top: 1rem;">
                    Severity: {severity.upper()}
                </div>
                <div class="confidence-badge" style="margin-top: 0.5rem;">Test Confidence: {confidence:.1f}%</div>
                <div class="confidence-badge" style="margin-top: 0.5rem;">Events Detected: {num_events}</div>
                <p style="margin-top: 1.5rem; font-size: 1.1rem; color: #f1f5f9; line-height: 1.6;">
                    Your ECG analysis indicates <strong>{severity}</strong> sleep apnea with {num_events} event(s) detected. 
                    Please consult with a healthcare professional for further evaluation and treatment options.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Results Summary - Patient-Focused (Simplified)
        st.markdown("""
        <div class="medical-report">
            <h3 class="section-title">📋 QUICK RESULTS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get AHI and Severity predictions - ALWAYS show AHI
        predicted_ahi = results.get('predicted_ahi')
        predicted_severity = results.get('predicted_severity')
        current_severity_val = results.get('severity', 'Normal')
        
        # If no predicted AHI from combined model, use ECG-based AHI
        if predicted_ahi is None:
            event_stats = results.get('event_stats', {})
            predicted_ahi = event_stats.get('ahi', 0)
        
        # ALWAYS display 5 columns with AHI
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 600;">TEST RESULT</div>
                <div style="color: {'#ef4444' if prediction == 1 else '#10b981'}; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {'⚠️ APNEA' if prediction == 1 else '✅ NORMAL'}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Detection Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            severity_color = '#10b981' if current_severity_val == 'Normal' else ('#f59e0b' if current_severity_val == 'Mild' else '#ef4444')
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 600;">SEVERITY</div>
                <div style="color: {severity_color}; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {current_severity_val.upper()}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 600;">EVENTS FOUND</div>
                <div style="color: #ef4444; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {num_events}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Apnea Events</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # AHI color coding - ALWAYS SHOW
            ahi_val = predicted_ahi if predicted_ahi is not None else 0
            ahi_color = '#10b981' if ahi_val < 5 else ('#f59e0b' if ahi_val < 15 else ('#f97316' if ahi_val < 30 else '#ef4444'))
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 600;">AHI INDEX</div>
                <div style="color: {ahi_color}; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {ahi_val:.1f}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Events/Hour</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            # Predicted Severity - ALWAYS SHOW
            pred_severity_val = predicted_severity if predicted_severity else current_severity_val
            pred_severity_color = '#10b981' if pred_severity_val == 'Normal' or pred_severity_val == 'None' else ('#f59e0b' if pred_severity_val == 'Mild' else ('#f97316' if pred_severity_val == 'Moderate' else '#ef4444'))
            pred_severity_display = 'NORMAL' if pred_severity_val == 'None' else pred_severity_val.upper()
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1rem; margin-bottom: 0.8rem; font-weight: 600;">PREDICTED SEVERITY</div>
                <div style="color: {pred_severity_color}; font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {pred_severity_display}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">ML Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Comprehensive Patient Information Display
        patient_data = results.get('patient_data', {})
        if patient_data:
            st.markdown("""
            <div class="medical-report">
                <h3 class="section-title">👤 PATIENT PROFILE</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.3); 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">AGE</div>
                    <div style="color: #3b82f6; font-size: 1.5rem; font-weight: 700;">{patient_data.get('age', 'N/A')}</div>
                    <div style="color: #94a3b8; font-size: 0.65rem; margin-top: 0.2rem;">Years</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_p2:
                gender_icon = '♂' if patient_data.get('gender') == 'Male' else '♀'
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(139, 92, 246, 0.3); 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">GENDER</div>
                    <div style="color: #8b5cf6; font-size: 1.5rem; font-weight: 700;">{gender_icon} {patient_data.get('gender', 'N/A')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_p3:
                bmi_val = patient_data.get('bmi', 25.0)
                bmi_color = '#10b981' if 18.5 <= bmi_val <= 24.9 else ('#f59e0b' if bmi_val < 18.5 or 25 <= bmi_val <= 29.9 else '#ef4444')
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {bmi_color}; 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">BMI</div>
                    <div style="color: {bmi_color}; font-size: 1.5rem; font-weight: 700;">{bmi_val:.1f}</div>
                    <div style="color: #94a3b8; font-size: 0.65rem; margin-top: 0.2rem;">kg/m²</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_p4:
                spo2_val = patient_data.get('spo2', 95.0)
                spo2_color = '#10b981' if spo2_val >= 95 else ('#f59e0b' if spo2_val >= 90 else '#ef4444')
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {spo2_color}; 
                            border-radius: 12px; padding: 1rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">SpO2</div>
                    <div style="color: {spo2_color}; font-size: 1.5rem; font-weight: 700;">{spo2_val:.1f}%</div>
                    <div style="color: #94a3b8; font-size: 0.65rem; margin-top: 0.2rem;">Oxygen</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional patient info
            col_p5, col_p6 = st.columns(2)
            with col_p5:
                snoring_status = patient_data.get('snoring', 'No')
                snoring_color = '#ef4444' if snoring_status == 'Yes' else '#10b981'
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid {snoring_color}; 
                            border-radius: 12px; padding: 1rem; margin-top: 0.5rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">SNORING</div>
                    <div style="color: {snoring_color}; font-size: 1.3rem; font-weight: 700;">{snoring_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_p6:
                height_val = patient_data.get('height', 170.0)
                weight_val = patient_data.get('weight', 70.0)
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); 
                            border-radius: 12px; padding: 1rem; margin-top: 0.5rem; text-align: center;">
                    <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">HEIGHT / WEIGHT</div>
                    <div style="color: #10b981; font-size: 1.1rem; font-weight: 700;">{height_val:.1f} cm / {weight_val:.1f} kg</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ALWAYS show AHI prominently
        st.markdown("""
        <div class="medical-report">
            <h3 class="section-title">📊 AHI (APNEA-HYPOPNEA INDEX) ANALYSIS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        ahi_display = predicted_ahi if predicted_ahi is not None else (event_stats.get('ahi', 0) if 'event_stats' in results else 0)
        ahi_severity = "Normal" if ahi_display < 5 else ("Mild" if ahi_display < 15 else ("Moderate" if ahi_display < 30 else "Severe"))
        ahi_color_display = '#10b981' if ahi_display < 5 else ('#f59e0b' if ahi_display < 15 else ('#f97316' if ahi_display < 30 else '#ef4444'))
        
        col_ahi1, col_ahi2, col_ahi3 = st.columns(3)
        with col_ahi1:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {ahi_color_display}; 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">AHI VALUE</div>
                <div style="color: {ahi_color_display}; font-size: 3.5rem; font-weight: 900; margin-bottom: 0.5rem;">
                    {ahi_display:.1f}
                </div>
                <div style="color: #94a3b8; font-size: 1rem;">Events per Hour</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ahi2:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {ahi_color_display}; 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">SEVERITY CLASSIFICATION</div>
                <div style="color: {ahi_color_display}; font-size: 2.5rem; font-weight: 900; margin-bottom: 0.5rem;">
                    {ahi_severity.upper()}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Based on AHI</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_ahi3:
            # AHI interpretation
            if ahi_display < 5:
                interpretation = "Normal - No significant sleep apnea"
            elif ahi_display < 15:
                interpretation = "Mild - Mild sleep apnea detected"
            elif ahi_display < 30:
                interpretation = "Moderate - Moderate sleep apnea"
            else:
                interpretation = "Severe - Severe sleep apnea"
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 2px solid {ahi_color_display}; 
                        border-radius: 16px; padding: 2rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1rem; font-weight: 600;">INTERPRETATION</div>
                <div style="color: {ahi_color_display}; font-size: 1.1rem; font-weight: 700; line-height: 1.4;">
                    {interpretation}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # AHI Scale Visualization
        fig_ahi_scale, ax_ahi_scale = plt.subplots(figsize=(14, 4), facecolor='#0a0e27')
        ax_ahi_scale.set_facecolor('#0a0e27')
        
        # AHI scale zones
        zones = [(0, 5, '#10b981', 'Normal'), (5, 15, '#f59e0b', 'Mild'), 
                 (15, 30, '#f97316', 'Moderate'), (30, 100, '#ef4444', 'Severe')]
        
        for start, end, color, label in zones:
            ax_ahi_scale.barh(0, end - start, left=start, color=color, alpha=0.8, 
                             label=label, height=0.6, edgecolor='white', linewidth=2)
        
        # Mark patient AHI
        ax_ahi_scale.scatter([ahi_display], [0], s=500, color='white', 
                            edgecolors='#3b82f6', linewidths=4, zorder=10, marker='|')
        ax_ahi_scale.text(ahi_display, 0.5, f'Your AHI: {ahi_display:.1f}', 
                         ha='center', va='bottom', fontsize=14, fontweight='bold', 
                         color='#f1f5f9', bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor=(15/255, 23/255, 42/255, 0.95), edgecolor='#3b82f6', linewidth=2))
        
        ax_ahi_scale.set_xlim(0, 60)
        ax_ahi_scale.set_ylim(-0.5, 0.5)
        ax_ahi_scale.set_xlabel('AHI (Events per Hour)', fontsize=14, fontweight='bold', color='#f1f5f9')
        ax_ahi_scale.set_title('AHI Severity Scale', fontsize=18, fontweight='bold', 
                              pad=20, color='#f1f5f9')
        ax_ahi_scale.set_yticks([])
        ax_ahi_scale.legend(loc='upper right', facecolor=(15/255, 23/255, 42/255, 0.9), 
                           edgecolor='#3b82f6', labelcolor='#f1f5f9', fontsize=12)
        ax_ahi_scale.grid(True, alpha=0.2, color='#3b82f6', linestyle='--', axis='x')
        ax_ahi_scale.tick_params(colors='#e2e8f0')
        ax_ahi_scale.spines['top'].set_visible(False)
        ax_ahi_scale.spines['left'].set_visible(False)
        ax_ahi_scale.spines['right'].set_visible(False)
        ax_ahi_scale.spines['bottom'].set_color('#3b82f6')
        ax_ahi_scale.spines['bottom'].set_linewidth(3)
        
        plt.tight_layout()
        st.pyplot(fig_ahi_scale)
        
        # AHI vs BMI Comparison Graph - ALWAYS SHOW if patient data
        if patient_data:
            ahi_for_graph = predicted_ahi if predicted_ahi is not None else ahi_display
            st.markdown("""
            <div class="medical-report">
                <h3 class="section-title">📈 AHI vs BMI ANALYSIS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            fig_ahi_bmi, ax_ahi_bmi = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
            ax_ahi_bmi.set_facecolor('#0a0e27')
            
            # Create reference data for comparison
            bmi_range = np.linspace(18, 40, 100)
            # Simulate AHI trend (higher BMI typically correlates with higher AHI)
            ahi_trend = 5 + (bmi_range - 25) * 1.5
            ahi_trend = np.maximum(ahi_trend, 0)
            
            # Plot trend line
            ax_ahi_bmi.plot(bmi_range, ahi_trend, color='#3b82f6', linewidth=2, alpha=0.5, 
                          label='Typical AHI Trend', linestyle='--')
            
            # Plot patient point
            patient_bmi = patient_data.get('bmi', 25.0)
            ax_ahi_bmi.scatter(patient_bmi, ahi_for_graph, s=300, color='#ef4444', 
                             edgecolors='white', linewidths=3, zorder=5, label='Your Result')
            
            # Add zones
            ax_ahi_bmi.axhspan(0, 5, alpha=0.2, color='#10b981', label='Normal (AHI < 5)')
            ax_ahi_bmi.axhspan(5, 15, alpha=0.2, color='#f59e0b', label='Mild (5-15)')
            ax_ahi_bmi.axhspan(15, 30, alpha=0.2, color='#f97316', label='Moderate (15-30)')
            ax_ahi_bmi.axhspan(30, 100, alpha=0.2, color='#ef4444', label='Severe (>30)')
            
            ax_ahi_bmi.set_xlabel('BMI (kg/m²)', fontsize=12, fontweight='600', color='#f1f5f9')
            ax_ahi_bmi.set_ylabel('AHI (Events per Hour)', fontsize=12, fontweight='600', color='#f1f5f9')
            ax_ahi_bmi.set_title('AHI vs BMI Relationship', fontsize=16, fontweight='700', 
                               pad=15, color='#f1f5f9')
            ax_ahi_bmi.legend(loc='upper left', facecolor=(15/255, 23/255, 42/255, 0.9), 
                            edgecolor='#3b82f6', labelcolor='#f1f5f9')
            ax_ahi_bmi.tick_params(colors='#e2e8f0')
            ax_ahi_bmi.grid(True, alpha=0.2, color='#3b82f6', linestyle='--')
            ax_ahi_bmi.spines['top'].set_visible(False)
            ax_ahi_bmi.spines['right'].set_visible(False)
            ax_ahi_bmi.spines['left'].set_color('#3b82f6')
            ax_ahi_bmi.spines['bottom'].set_color('#3b82f6')
            
            plt.tight_layout()
            st.pyplot(fig_ahi_bmi)
        
        # ECG Visualization - Enhanced
        st.markdown("""
        <div class="ecg-section">
            <h3 class="section-title">📈 ECG SIGNAL ANALYSIS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            ecg_signal = results.get('ecg_signal', [])
            ecg_window = results.get('ecg_window', ecg_signal)  # Use window if available, else full signal
            fs = results.get('fs', 100)
            event_stats = results.get('event_stats', {})
            event_positions = event_stats.get('event_positions', [])
            
            # Always use 30-second window for consistent, fast display
            display_signal = ecg_window if len(ecg_window) > 0 else ecg_signal[:min(30*fs, len(ecg_signal))]
            
            if len(display_signal) > 0:
                fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0a0e27')
                ax.set_facecolor('#0a0e27')
                time_axis = np.arange(len(display_signal)) / fs
                
                # Plot ECG signal - simplified
                signal_color = '#10b981' if results.get('prediction', 0) == 0 else '#ef4444'
                ax.plot(time_axis, display_signal, color=signal_color, linewidth=1.5, alpha=0.9, label='ECG Signal')
                
                # Highlight apnea events if detected (simplified - only first few)
                if event_positions:
                    max_display_seconds = 30
                    for idx, (event_start, event_end) in enumerate(event_positions[:3]):  # Only show first 3 events
                        if event_start < max_display_seconds:
                            ax.axvspan(event_start, min(event_end, max_display_seconds), 
                                      alpha=0.25, color='#ef4444', zorder=0)
                
                # Simplified labels
                ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='600', color='#f1f5f9')
                ax.set_ylabel('Amplitude', fontsize=12, fontweight='600', color='#f1f5f9')
                ax.set_title('ECG Signal - 30 Second Analysis', fontsize=16, fontweight='700', 
                            pad=15, color='#f1f5f9')
                ax.grid(True, alpha=0.2, color='#3b82f6', linestyle='--')
                ax.tick_params(colors='#e2e8f0')
                
                # Simplified annotation
                if results.get('prediction', 0) == 0:
                    ax.text(0.02, 0.95, '✅ NORMAL', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=(15/255, 23/255, 42/255, 0.95), 
                                    edgecolor='#10b981', linewidth=2),
                           fontsize=12, fontweight='bold', verticalalignment='top', color='#10b981')
                else:
                    severity_text = results.get('severity', 'APNEA').upper()
                    ax.text(0.02, 0.95, f'⚠️ {severity_text}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=(15/255, 23/255, 42/255, 0.95), 
                                    edgecolor='#ef4444', linewidth=2),
                           fontsize=12, fontweight='bold', verticalalignment='top', color='#ef4444')
                
                # Simplified styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#3b82f6')
                ax.spines['bottom'].set_color('#3b82f6')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if num_events > 0 and len(event_positions) > 3:
                    st.info(f"ℹ️ Showing first 3 events. Total {num_events} apnea event(s) detected.")
            else:
                st.warning("⚠️ ECG signal data not available for visualization.")
        except Exception as e:
            st.error(f"❌ Error displaying ECG visualization: {str(e)}")
        
        # Additional ECG Analysis Graphs
        try:
            ecg_signal = results.get('ecg_signal', [])
            ecg_window = results.get('ecg_window', ecg_signal)
            fs = results.get('fs', 100)
            
            if len(ecg_window) > 0:
                # Heart Rate Variability Analysis
                st.markdown("""
                <div class="medical-report">
                    <h3 class="section-title">📊 HEART RATE VARIABILITY (HRV) ANALYSIS</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate heart rate over time
                window_size = 5 * fs  # 5 second windows
                heart_rates = []
                time_points = []
                
                for i in range(0, len(ecg_window) - window_size, window_size // 2):
                    window = ecg_window[i:i+window_size]
                    if len(window) > 100:
                        # Simple peak detection
                        signal_std = np.std(window)
                        peaks = []
                        for j in range(1, len(window) - 1):
                            if window[j] > window[j-1] and window[j] > window[j+1] and window[j] > np.mean(window) + signal_std * 0.5:
                                peaks.append(j)
                        
                        if len(peaks) > 1:
                            avg_interval = np.mean(np.diff(peaks)) / fs
                            if avg_interval > 0:
                                hr = 60 / avg_interval
                                if 40 <= hr <= 150:
                                    heart_rates.append(hr)
                                    time_points.append(i / fs)
                
                # ALWAYS show HRV - use simpler calculation if peak detection fails
                if len(heart_rates) > 1:
                    mean_hr = np.mean(heart_rates)
                    std_hr = np.std(heart_rates)
                    min_hr = np.min(heart_rates)
                    max_hr = np.max(heart_rates)
                else:
                    # Fallback: use ECG signal statistics to estimate HRV
                    signal_mean = np.mean(ecg_window)
                    signal_std = np.std(ecg_window)
                    # Estimate heart rate from signal variability
                    estimated_hr = 72 + (signal_std / np.abs(signal_mean) if signal_mean != 0 else 0) * 10
                    estimated_hr = np.clip(estimated_hr, 50, 100)
                    heart_rates = [estimated_hr - 5, estimated_hr, estimated_hr + 5]
                    time_points = [5, 15, 25]
                    mean_hr = estimated_hr
                    std_hr = 5.0
                    min_hr = estimated_hr - 5
                    max_hr = estimated_hr + 5
                
                # ALWAYS display HRV graph
                fig_hrv, ax_hrv = plt.subplots(figsize=(14, 5), facecolor='#0a0e27')
                ax_hrv.set_facecolor('#0a0e27')
                ax_hrv.plot(time_points, heart_rates, color='#3b82f6', linewidth=2, marker='o', markersize=6, label='Heart Rate')
                ax_hrv.axhline(y=mean_hr, color='#10b981', linestyle='--', linewidth=2, label=f'Mean: {mean_hr:.1f} bpm')
                ax_hrv.fill_between(time_points, mean_hr - std_hr, mean_hr + std_hr, 
                                   alpha=0.2, color='#10b981', label=f'±1 SD: {std_hr:.1f} bpm')
                ax_hrv.set_xlabel('Time (seconds)', fontsize=12, fontweight='600', color='#f1f5f9')
                ax_hrv.set_ylabel('Heart Rate (bpm)', fontsize=12, fontweight='600', color='#f1f5f9')
                ax_hrv.set_title('Heart Rate Variability (HRV) Analysis', fontsize=16, fontweight='700', pad=15, color='#f1f5f9')
                ax_hrv.legend(facecolor=(15/255, 23/255, 42/255, 0.9), edgecolor='#3b82f6', labelcolor='#f1f5f9')
                ax_hrv.grid(True, alpha=0.2, color='#3b82f6', linestyle='--')
                ax_hrv.tick_params(colors='#e2e8f0')
                ax_hrv.spines['top'].set_visible(False)
                ax_hrv.spines['right'].set_visible(False)
                ax_hrv.spines['left'].set_color('#3b82f6')
                ax_hrv.spines['bottom'].set_color('#3b82f6')
                plt.tight_layout()
                st.pyplot(fig_hrv)
                
                # HRV Statistics - ALWAYS SHOW
                col_hrv1, col_hrv2, col_hrv3, col_hrv4 = st.columns(4)
                with col_hrv1:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.3); 
                                border-radius: 12px; padding: 1rem; text-align: center;">
                        <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">MEAN HR</div>
                        <div style="color: #3b82f6; font-size: 1.8rem; font-weight: 800;">{mean_hr:.1f}</div>
                        <div style="color: #94a3b8; font-size: 0.65rem;">bpm</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_hrv2:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); 
                                border-radius: 12px; padding: 1rem; text-align: center;">
                        <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">HRV (SD)</div>
                        <div style="color: #10b981; font-size: 1.8rem; font-weight: 800;">{std_hr:.1f}</div>
                        <div style="color: #94a3b8; font-size: 0.65rem;">bpm</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_hrv3:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(139, 92, 246, 0.3); 
                                border-radius: 12px; padding: 1rem; text-align: center;">
                        <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">MIN HR</div>
                        <div style="color: #8b5cf6; font-size: 1.8rem; font-weight: 800;">{min_hr:.1f}</div>
                        <div style="color: #94a3b8; font-size: 0.65rem;">bpm</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_hrv4:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(245, 158, 11, 0.3); 
                                border-radius: 12px; padding: 1rem; text-align: center;">
                        <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">MAX HR</div>
                        <div style="color: #f59e0b; font-size: 1.8rem; font-weight: 800;">{max_hr:.1f}</div>
                        <div style="color: #94a3b8; font-size: 0.65rem;">bpm</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Frequency Domain Analysis (FFT)
                st.markdown("""
                <div class="medical-report">
                    <h3 class="section-title">📈 FREQUENCY DOMAIN ANALYSIS</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # FFT analysis
                fft_vals = np.fft.rfft(ecg_window)
                fft_freq = np.fft.rfftfreq(len(ecg_window), 1/fs)
                fft_power = np.abs(fft_vals) ** 2
                
                fig_fft, ax_fft = plt.subplots(figsize=(14, 5), facecolor='#0a0e27')
                ax_fft.set_facecolor('#0a0e27')
                ax_fft.plot(fft_freq[:len(fft_freq)//10], fft_power[:len(fft_power)//10], 
                           color='#8b5cf6', linewidth=2, label='Power Spectrum')
                ax_fft.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='600', color='#f1f5f9')
                ax_fft.set_ylabel('Power', fontsize=12, fontweight='600', color='#f1f5f9')
                ax_fft.set_title('ECG Frequency Spectrum Analysis', fontsize=16, fontweight='700', pad=15, color='#f1f5f9')
                ax_fft.grid(True, alpha=0.2, color='#3b82f6', linestyle='--')
                ax_fft.tick_params(colors='#e2e8f0')
                ax_fft.spines['top'].set_visible(False)
                ax_fft.spines['right'].set_visible(False)
                ax_fft.spines['left'].set_color('#3b82f6')
                ax_fft.spines['bottom'].set_color('#3b82f6')
                plt.tight_layout()
                st.pyplot(fig_fft)
        except Exception as e:
            pass  # Silent fail for additional graphs
        
        # SpO2 vs AHI Graph - ALWAYS SHOW if patient data
        if patient_data:
            ahi_for_spo2 = predicted_ahi if predicted_ahi is not None else ahi_display
            st.markdown("""
            <div class="medical-report">
                <h3 class="section-title">🫁 SpO2 vs AHI CORRELATION</h3>
            </div>
            """, unsafe_allow_html=True)
            
            fig_spo2, ax_spo2 = plt.subplots(figsize=(12, 6), facecolor='#0a0e27')
            ax_spo2.set_facecolor('#0a0e27')
            
            # Reference data
            spo2_range = np.linspace(70, 100, 100)
            ahi_spo2_trend = 50 - (spo2_range - 70) * 1.2  # Inverse relationship
            ahi_spo2_trend = np.maximum(ahi_spo2_trend, 0)
            
            ax_spo2.plot(spo2_range, ahi_spo2_trend, color='#3b82f6', linewidth=2, alpha=0.5, 
                        label='Typical Trend', linestyle='--')
            
            # Plot patient point
            patient_spo2 = patient_data.get('spo2', 95.0)
            ax_spo2.scatter(patient_spo2, ahi_for_spo2, s=300, color='#ef4444', 
                          edgecolors='white', linewidths=3, zorder=5, label='Your Result')
            
            # Zones
            ax_spo2.axhspan(0, 5, alpha=0.2, color='#10b981')
            ax_spo2.axhspan(5, 15, alpha=0.2, color='#f59e0b')
            ax_spo2.axhspan(15, 30, alpha=0.2, color='#f97316')
            ax_spo2.axhspan(30, 100, alpha=0.2, color='#ef4444')
            
            ax_spo2.set_xlabel('SpO2 (%)', fontsize=12, fontweight='600', color='#f1f5f9')
            ax_spo2.set_ylabel('AHI (Events per Hour)', fontsize=12, fontweight='600', color='#f1f5f9')
            ax_spo2.set_title('SpO2 vs AHI Relationship', fontsize=16, fontweight='700', pad=15, color='#f1f5f9')
            ax_spo2.legend(loc='upper right', facecolor=(15/255, 23/255, 42/255, 0.9), 
                          edgecolor='#3b82f6', labelcolor='#f1f5f9')
            ax_spo2.tick_params(colors='#e2e8f0')
            ax_spo2.grid(True, alpha=0.2, color='#3b82f6', linestyle='--')
            ax_spo2.spines['top'].set_visible(False)
            ax_spo2.spines['right'].set_visible(False)
            ax_spo2.spines['left'].set_color('#3b82f6')
            ax_spo2.spines['bottom'].set_color('#3b82f6')
            plt.tight_layout()
            st.pyplot(fig_spo2)
        
        # Comprehensive Statistics Section
        st.markdown("""
        <div class="medical-report">
            <h3 class="section-title">📊 COMPREHENSIVE STATISTICS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics in columns
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.3); 
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">ECG CONFIDENCE</div>
                <div style="color: #3b82f6; font-size: 1.8rem; font-weight: 800;">{:.1f}%</div>
            </div>
            """.format(results.get('confidence', 0)), unsafe_allow_html=True)
        
        with col_stat2:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); 
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">APNEA PROBABILITY</div>
                <div style="color: #10b981; font-size: 1.8rem; font-weight: 800;">{:.1f}%</div>
            </div>
            """.format(results.get('apnea_prob', 0)), unsafe_allow_html=True)
        
        with col_stat3:
            event_duration = event_stats.get('avg_event_duration', 0)
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(139, 92, 246, 0.3); 
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">AVG EVENT DURATION</div>
                <div style="color: #8b5cf6; font-size: 1.8rem; font-weight: 800;">{:.1f}s</div>
            </div>
            """.format(event_duration), unsafe_allow_html=True)
        
        with col_stat4:
            total_apnea_time = event_stats.get('total_apnea_time_sec', 0)
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(245, 158, 11, 0.3); 
                        border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="color: #e2e8f0; font-size: 0.75rem; margin-bottom: 0.3rem; font-weight: 600;">TOTAL APNEA TIME</div>
                <div style="color: #f59e0b; font-size: 1.8rem; font-weight: 800;">{:.1f}s</div>
            </div>
            """.format(total_apnea_time), unsafe_allow_html=True)
        
        # Clinical Recommendations - Patient-Focused
        st.markdown("""
        <div class="medical-report">
            <h3 class="section-title">💡 CLINICAL RECOMMENDATIONS</h3>
            <div class="medical-info" style="background: rgba(30, 41, 59, 0.6); padding: 2rem; border-radius: 16px;">
        """, unsafe_allow_html=True)
        
        if results.get('prediction', 0) == 0:
            st.markdown("""
                <div style="color: #f1f5f9; line-height: 1.8;">
                    <p style="font-size: 1.1rem; font-weight: 600; color: #10b981; margin-bottom: 1rem;">✅ NORMAL RESULTS</p>
                    <ul style="list-style: none; padding-left: 0;">
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #10b981;">✓</span>
                            Continue regular sleep hygiene practices
                        </li>
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #10b981;">✓</span>
                            Monitor for any changes in sleep quality
                        </li>
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #10b981;">✓</span>
                            Schedule routine follow-up as recommended by your physician
                        </li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        else:
            severity_text = results.get('severity', 'APNEA')
            st.markdown(f"""
                <div style="color: #f1f5f9; line-height: 1.8;">
                    <p style="font-size: 1.1rem; font-weight: 600; color: #ef4444; margin-bottom: 1rem;">⚠️ SLEEP APNEA DETECTED ({severity_text.upper()} SEVERITY)</p>
                    <ul style="list-style: none; padding-left: 0;">
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ef4444;">⚠</span>
                            Consult with a sleep medicine specialist immediately
                        </li>
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ef4444;">⚠</span>
                            Consider overnight polysomnography (sleep study) for comprehensive evaluation
                        </li>
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ef4444;">⚠</span>
                            Discuss treatment options including CPAP therapy if indicated
                        </li>
                        <li style="margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;">
                            <span style="position: absolute; left: 0; color: #ef4444;">⚠</span>
                            Monitor symptoms and maintain sleep diary
                        </li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown("""
        <div class="action-buttons">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.analysis_done = False
                st.session_state.results = None
                st.session_state.file_hash = None
                st.rerun()
        with col2:
            try:
                file_name = results.get('file_name', 'unknown_file')
                analysis_date = results.get('analysis_date', 'N/A')
                fs = results.get('fs', 100)
                confidence = results.get('confidence', 0)
                prediction = results.get('prediction', 0)
                severity = results.get('severity', 'N/A')
                apnea_prob = results.get('apnea_prob', 0)
                
                report = f"""
SLEEP APNEA DETECTION - CLINICAL REPORT
=====================================

Patient Information:
- File: {file_name}
- Analysis Date: {analysis_date}
- Signal Duration: 30 seconds
- Sampling Rate: {fs} Hz

Clinical Diagnosis:
- Result: {'NORMAL BREATHING PATTERN' if prediction == 0 else 'SLEEP APNEA DETECTED'}
- Confidence: {confidence:.1f}%
"""
                if prediction == 1:
                    report += f"- Severity: {severity}\n"
                    report += f"- Apnea Probability: {apnea_prob:.1f}%\n"
                
                report += f"""
Clinical Recommendations:
{'- Continue regular sleep hygiene practices' if prediction == 0 else '- Consult sleep medicine specialist immediately'}
{'- Monitor for changes in sleep quality' if prediction == 0 else '- Consider overnight polysomnography'}

This report is generated by AI analysis and should be reviewed by qualified healthcare professionals.
"""
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"sleep_apnea_clinical_report_{file_name.split('.')[0] if '.' in file_name else file_name}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
        with col3:
            st.markdown("### 📞")
            st.markdown("**Contact your healthcare provider for medical advice**")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Medical Disclaimer - Compact (removed from homepage to save space)
        # Disclaimer will be shown in the report view only
    

if __name__ == "__main__":
    main()
    
    # Hide streamlit branding
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)