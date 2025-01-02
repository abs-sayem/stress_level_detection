# **Stress Level Detection**
A Comparison Study of Statistical ML Model vs CNN Model vs Proposed (LSTM+CNN) Model
---

### **Aim of this Study:**
- Propose an optimal multi-modal deep learning approach for stress level detection.
- Compare my proposed model with a well stablished CNN model and one of two Statistical (Traditional) ML models.
### **Approach of this Study:**
- **A Suitable Dataset:** Tononto Emotional Speech Set (TESS) dataset includes almost all the common emotions of a human, emotionally rich and semantically neutral utterances and high audio quality data.
- **Suitable Models to Train:**
    - Statistical (ML) Models: I select SVM and Random Forest to train on my dataset and further choose best one.
    - CNN Model: StressCNN model is published in December 2021 for Stress Classification. It is simple yet powerful architecture for capturing spatio-temporal patterns from audio, robustness to noise, scalability, and adaptability to stress-specific acoustic features.
    - Proposed (LSTM+CNN) Model: Here I use a combination of lstm and cnn architecture.
    - `why this (lstm+cnn) combination?`
        - lstm is designed to handle sequential data, so it will ensure the sequential relationships over time. capturing temporal dependencies like - stress-related changes (e.g., in pitch, tone, or energy) are very crucial for stress level detection.
        - cnn can isolate local patterns and also very suitable to capture instantaneous features. (will be helpful because our audio files are (1-3) sec long).
        - lstm and cnn combination is flexible for multimodal extension where lstm will provide context that pure CNNs might miss and will create a base for incorporating modalities (e.g., physiological signals, text), making it easier to extend the model for future applications.
    **Combination of Approaches:**
    - Train all the three models on Feature Dataset.
    - Train all the three models on Raw Audio Dataset.
    - Train all the three models on Combined Audio and Feature Dataset.
### **Dataset**
---
#### **TESS (Tononto Emotional Speech Set) Dataset**
- The primary dataset is collected from kaggle [dataset link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

**Key Features:**
- Includes recordings from two female native English speakers: aged 26 and 64 years.
- The dataset captures seven emotions: `Neutral`, `Angry`, `Happy`, `Sad`, `Disgust`, `Fear`, and `Pleasant Surprise`.
- Each emotion is expressed using 200 target words, producing emotionally rich and semantically neutral utterances.
- The recordings are provided in 16-bit WAV format with a sampling rate of 44.1 kHz, ensures high audio quality suitable for acoustic analysis.
#### **Making of Feature Dataset**
- For comparison with cnn and proposed (lstm+cnn) model a feature dataset is needed for traditional machile learning models. This dataset is used to train all the models for one of out three sections of comparison.
- `librosa` and `parselmouth` both are python libraries used to extract features from audio file.
    - librosa extracted features like - Pitch, Intensity, Frequency, Amplitude, MFCCs, Spectral - (Centroid, Bandwidth, Rolloff, Flux), Energy, ZCR, Tempo, Formants.
    - perselmouth extracted features like - Jitter and Shimmer [These valuses were null, they were removed]

**Dataset with all Extracted Features**
![Dataset with all extracted features](images/all_extracted_features.jpg)
**Final Dataset for Training**
![Final Dataset for Training](images/final_features_for_train.jpg)

Comparison study of OpenSMILE+DT, Pretrained CNN and LSTM+CNN (proposed) on TESS dataset for research purpose
