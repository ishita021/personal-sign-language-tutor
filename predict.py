"""Real-time, smoothed sign prediction from normalized hand landmarks."""
from collections import deque
from pathlib import Path
import time
import cv2, joblib, numpy as np
from hand_detection import HandDetector, LandmarkNormalizer, FeatureExtractor, landmarks_to_array

class ModelLoader:
    def __init__(self, directory=Path("models")): self.directory = Path(directory)
    def load(self):
        model, encoder = self.directory / "gesture_model.pkl", self.directory / "label_encoder.pkl"
        if not model.exists() or not encoder.exists(): raise FileNotFoundError("Train a model before prediction.")
        return joblib.load(model), joblib.load(encoder)

class PredictionSmoother:
    def __init__(self, size=7): self.values = deque(maxlen=size)
    def update(self, label, confidence):
        self.values.append((label, confidence)); labels = [v[0] for v in self.values]
        winner = max(set(labels), key=labels.count); scores = [v[1] for v in self.values if v[0] == winner]
        return winner, float(np.mean(scores))

class GesturePredictor:
    def run(self):
        try: model, encoder = ModelLoader().load()
        except (FileNotFoundError, Exception) as error: print("ERROR:", error); return
        camera = cv2.VideoCapture(0)
        if not camera.isOpened(): print("ERROR: Cannot open webcam."); return
        detector, normalizer, extractor, smoother = HandDetector(max_hands=1), LandmarkNormalizer(), FeatureExtractor(), PredictionSmoother(); previous=time.perf_counter()
        try:
            while True:
                ok, frame=camera.read()
                if not ok: break
                frame=cv2.flip(frame,1); shown, landmarks, found=detector.find_hands(frame); text="No Hand Detected"
                if found:
                    vector=extractor.extract(normalizer.normalize(landmarks_to_array(landmarks)))
                    if vector is not None:
                        probabilities=model.predict_proba([vector])[0]; index=int(np.argmax(probabilities)); label=encoder.inverse_transform([model.classes_[index]])[0]
                        stable, confidence=smoother.update(label, probabilities[index]); text=f"{stable}: {confidence:.0%}"
                now=time.perf_counter(); fps=1/max(now-previous,1e-9); previous=now
                cv2.putText(shown,text,(10,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,255,0),2); cv2.putText(shown,f"FPS: {fps:.1f}",(10,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                cv2.imshow("Sign Prediction",shown)
                if cv2.waitKey(1)&255==27: break
        finally: camera.release(); detector.close(); cv2.destroyAllWindows()

if __name__ == "__main__": GesturePredictor().run()
