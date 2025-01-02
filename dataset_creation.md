#### **TESS (Tononto Emotional Speech Set) Dataset**
- The primary dataset is collected from kaggle [dataset link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

**Key Features:**
- Includes recordings from two female native English speakers: aged 26 and 64 years.
- The dataset captures seven emotions: `Neutral`, `Angry`, `Happy`, `Sad`, `Disgust`, `Fear`, and `Pleasant Surprise`.
- Each emotion is expressed using 200 target words, producing emotionally rich and semantically neutral utterances.
- The recordings are provided in 16-bit WAV format with a sampling rate of 44.1 kHz, ensures high audio quality suitable for acoustic analysis.

**Making of Feature Dataset**
- For comparison with cnn and proposed (lstm+cnn) model a feature dataset is needed for traditional machile learning models. This dataset is used to train all the models for one of out three sections of comparison.
- `librosa` and `parselmouth` both are python libraries used to extract features from audio file.
    - librosa extracted features like - Pitch, Intensity, Frequency, Amplitude, MFCCs, Spectral - (Centroid, Bandwidth, Rolloff, Flux), Energy, ZCR, Tempo, Formants.
    - perselmouth extracted features like - Jitter and Shimmer [These valuses were null, they were removed]

**Dataset with all Extracted Features**
![Dataset with all extracted features](images/all_extracted_features.jpg)
**Final Dataset for Training**
![Final Dataset for Training](images/final_features_for_train.jpg)
