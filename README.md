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

## Hand landmark feature extraction

Run `python hand_detection.py` to show 21 landmark IDs and create a feature
vector for every detected hand. Raw MediaPipe coordinates include both the hand
pose and its position in the camera image. The pipeline subtracts landmark 0
(the wrist) from all landmarks, placing the wrist at `(0, 0, 0)`, then divides
by the largest wrist-to-landmark distance. This reduces the effects of screen
location, resolution, hand size, camera distance, and small movements.

Each hand becomes a fixed 63-value vector in `[x0, y0, z0, ..., x20, y20, z20]`
order. A later sign-language model can train on these normalized pose features
instead of learning irrelevant screen position. The webcam overlay shows IDs,
hand count, feature length, and FPS; raw and normalized values print once per
second for debugging.

## Tutor dashboard

Run `streamlit run app.py` for Dashboard, Tutor, Learning, Quiz, and Settings
modes. Progress and preferences are saved locally under `data/`. The dashboard
works offline; voice guidance uses the browser's speech engine when enabled.
Random Forest was selected because it is robust for compact landmark vectors,
needs little tuning, and provides confidence probabilities for feedback.

## Hand landmark feature extraction

Run the real-time landmark pipeline with:

```
python hand_detection.py
```

For every detected hand, MediaPipe provides 21 raw `(x, y, z)` landmarks. Raw
coordinates describe both the pose and where the hand happens to be in the
camera frame, so they are not ideal ML inputs. The pipeline subtracts landmark
`0` (the wrist) from every landmark, making the wrist `(0, 0, 0)`, then divides
by the largest wrist-to-landmark distance. This makes features substantially
less sensitive to image position, resolution, camera distance, hand size, and
small movements.

The resulting fixed-length feature vector contains 63 values in landmark-major
order: `[x0, y0, z0, x1, y1, z1, ... x20, y20, z20]`. Later, a sign-language
model can learn pose patterns from these comparable vectors without being
distracted by the hand's location in the image. The script draws landmark IDs,
detected-hand count, feature length, and FPS on the webcam window, and prints
raw and normalized coordinates once per second for debugging.
