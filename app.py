"""Virtual Sign Language Tutor hackathon dashboard. Run: streamlit run app.py"""
from __future__ import annotations

import random
import streamlit as st
from tutor_core import APP_VERSION, available_labels, feedback, load_progress, load_settings, record, save_settings

st.set_page_config(page_title="Virtual Sign Language Tutor", page_icon="🤟", layout="wide")

def speak(message: str, enabled: bool) -> None:
    if enabled:
        st.components.v1.html(f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance({message!r}))</script>", height=0)

def choose_target(labels):
    if "target" not in st.session_state: st.session_state.target = random.choice(labels)
    return st.session_state.target

def next_target(labels): st.session_state.target = random.choice(labels)

settings = load_settings(); progress = load_progress(); labels = available_labels()
st.sidebar.title("🤟 Sign Tutor")
page = st.sidebar.radio("Navigate", ["Dashboard", "Tutor Mode", "Learning Mode", "Quiz Mode", "Settings"])
st.sidebar.caption(f"v{APP_VERSION} • ESC closes webcam tools")

if page == "Dashboard":
    st.title("Welcome back")
    a,b,c,d = st.columns(4)
    a.metric("Today's practice", progress.attempted); b.metric("Overall accuracy", f"{progress.accuracy:.0f}%")
    c.metric("Best score", progress.best_score); d.metric("Signs learned", len({x['target'] for x in progress.activities if x['correct']}))
    st.subheader("Recent activity")
    st.dataframe(progress.activities or [{"time":"—", "target":"No practice yet", "prediction":"—", "correct":False}], use_container_width=True, hide_index=True)
    st.info("Use Dataset Collection and Train Model scripts first for live webcam predictions. Tutor modes remain usable with manual review.")
elif page in ("Tutor Mode", "Learning Mode"):
    target = choose_target(labels) if page == "Tutor Mode" else st.selectbox("Choose a sign", labels)
    st.title("Tutor Mode" if page == "Tutor Mode" else "Learning Mode")
    left, right = st.columns([2,1]); left.markdown(f"### Show the sign for: **{target}**"); left.info("Webcam prediction is available through `python predict.py`. Select your observed prediction below to log practice in this dashboard.")
    predicted = right.selectbox("Prediction", labels, key=f"prediction-{page}"); confidence = right.slider("Confidence", 0.0, 1.0, .80, .01)
    if right.button("Check answer"):
        correct = record(progress, target, predicted, confidence); message = "✅ Correct!" if correct else "❌ Incorrect — try again."
        st.success(message) if correct else st.warning(message); st.write(feedback(target,predicted,confidence)); speak("Excellent" if correct else "Please try again", settings["voice"])
        if correct and page == "Tutor Mode": next_target(labels)
elif page == "Quiz Mode":
    if "quiz" not in st.session_state: st.session_state.quiz = random.choices(labels, k=10); st.session_state.q = 0; st.session_state.score = 0
    q = st.session_state.q
    if q >= 10:
        score=st.session_state.score; stars="⭐"*(3 if score>=80 else 2 if score>=50 else 1)
        st.title("Quiz complete"); st.metric("Final score", f"{score}/100"); st.write(stars, "Excellent" if score>=80 else "Good" if score>=50 else "Needs Practice")
        if st.button("New quiz"): del st.session_state.quiz, st.session_state.q, st.session_state.score; st.rerun()
    else:
        target=st.session_state.quiz[q]; st.title(f"Question {q+1} of 10"); st.metric("Score", st.session_state.score)
        predicted=st.selectbox("Your predicted sign", labels); confidence=st.slider("Confidence",0.,1.,.8,.01)
        if st.button("Submit"):
            correct=record(progress,target,predicted,confidence,st.session_state.score+(10 if target==predicted and confidence>=.65 else 0)); st.session_state.score += 10 if correct else 0; st.session_state.q += 1; st.rerun()
else:
    st.title("Settings")
    settings["voice"] = st.toggle("Voice guidance", settings["voice"]); settings["show_fps"] = st.toggle("Show FPS", settings["show_fps"])
    settings["dark_mode"] = st.toggle("Dark mode", settings["dark_mode"]); settings["camera"] = st.number_input("Camera index", 0, 10, settings["camera"])
    settings["threshold"] = st.slider("Prediction confidence threshold", .1, 1., settings["threshold"], .05)
    if st.button("Save settings"): save_settings(settings); st.success("Settings saved locally.")
