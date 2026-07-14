# Virtual Sign Language Tutor

A Python-based AI application that uses your webcam and machine learning to teach and recognize sign language in real time.

## Project Structure

```
Virtual-Sign-Language-Tutor/
│
├── dataset/          # Raw image data for each sign label
├── models/           # Saved trained ML models
├── collected_data/   # CSV/numpy files of extracted hand landmarks
├── static/           # Static assets (images, icons, etc.)
│
├── app.py            # Main Streamlit web application
├── collect_data.py   # Webcam data collection script
├── train_model.py    # Model training script
├── predict.py        # Real-time sign prediction script
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## Setup

1. Create and activate the virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dependencies

| Package | Purpose |
|---|---|
| opencv-python | Webcam access and image processing |
| mediapipe | Hand landmark detection |
| numpy | Numerical operations on landmark arrays |
| pandas | Storing and managing collected landmark data |
| scikit-learn | Training the sign classification model |
| streamlit | Interactive web UI for the tutor app |
| joblib | Saving and loading trained models |

## Usage

```
# Run the main app
streamlit run app.py
```
