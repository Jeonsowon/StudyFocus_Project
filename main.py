import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from PIL import Image
import simpleaudio as sa

st.set_page_config(page_title="강의 집중도 측정", layout="wide")

# 상태 저장
if "score_log" not in st.session_state:
    st.session_state.score_log = []
if "warning_log" not in st.session_state:
    st.session_state.warning_log = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "calibration_sum" not in st.session_state:
    st.session_state.calibration_sum = [0, 0]
if "calibrated_center" not in st.session_state:
    st.session_state.calibrated_center = None

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
CALIBRATION_FRAMES = 60
EAR_THRESHOLD = 0.2
EYE_CLOSE_DURATION = 3
HEAD_MOVE_DURATION = 10
HEAD_MOVE_THRESHOLD_X = 60
HEAD_MOVE_THRESHOLD_Y = 40

head_off_start = None
eyes_closed_start = None
head_moved = False
eyes_closed = False
head_moved_timer = 0
eyes_closed_timer = 0


def calculate_ear(landmarks, eye_indices, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * np.linalg.norm(p[0] - p[3]))
    return ear


def play_alert():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
        wave_obj.play()
    except:
        pass

st.title("🎓 인터넷 강의 집중도 측정기")
left_col, right_col = st.columns([1, 1])

FRAME_WINDOW = left_col.image([])
st_focused = right_col.empty()
st_timer = right_col.empty()
st_chart = right_col.empty()
st_log = right_col.empty()

stop = st.sidebar.button("🛑 세션 종료 및 저장", key="stop_button")

cap = cv2.VideoCapture(0)
focus_score = 0
score_history = []

while cap.isOpened() and not stop:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = "Detecting..."
    head_moved = False
    eyes_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose = landmarks[1]
            nx, ny = nose.x * w, nose.y * h

            if st.session_state.calibrated_center is None and st.session_state.frame_count < CALIBRATION_FRAMES:
                st.session_state.calibration_sum[0] += nx
                st.session_state.calibration_sum[1] += ny
                st.session_state.frame_count += 1
                status_text = f"📐 기준점 설정 중... ({st.session_state.frame_count}/{CALIBRATION_FRAMES})"
                if st.session_state.frame_count == CALIBRATION_FRAMES:
                    cx = st.session_state.calibration_sum[0] / CALIBRATION_FRAMES
                    cy = st.session_state.calibration_sum[1] / CALIBRATION_FRAMES
                    st.session_state.calibrated_center = (cx, cy)
                continue

            # 기준점 기준 고개 이탈 감지
            if st.session_state.calibrated_center:
                cx, cy = st.session_state.calibrated_center
                dx = abs(cx - nx)
                dy = abs(cy - ny)
                if dx > HEAD_MOVE_THRESHOLD_X or dy > HEAD_MOVE_THRESHOLD_Y:
                    if not head_off_start:
                        head_off_start = time.time()
                    elif time.time() - head_off_start >= HEAD_MOVE_DURATION:
                        head_moved = True
                        if head_moved_timer == 0 or time.time() - head_moved_timer > 2:
                            play_alert()
                            st.session_state.warning_log.append((time.strftime("%H:%M:%S"), "고개 이탈 경고"))
                            head_moved_timer = time.time()
                else:
                    head_off_start = None

            # 눈 감김 감지
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < EAR_THRESHOLD:
                if not eyes_closed_start:
                    eyes_closed_start = time.time()
                elif time.time() - eyes_closed_start >= EYE_CLOSE_DURATION:
                    eyes_closed = True
                    if eyes_closed_timer == 0 or time.time() - eyes_closed_timer > 2:
                        play_alert()
                        st.session_state.warning_log.append((time.strftime("%H:%M:%S"), "눈 감김 경고"))
                        eyes_closed_timer = time.time()
            else:
                eyes_closed_start = None

    if head_moved:
        status_text = "🧠 Head Moved"
    elif eyes_closed:
        status_text = "😴 Eyes Closed"
    elif st.session_state.frame_count >= CALIBRATION_FRAMES:
        status_text = "✅ Focused"
        focus_score += 1

    st.session_state.score_log.append(focus_score)
    elapsed = int(time.time() - st.session_state.start_time)

    FRAME_WINDOW.image(img, channels="BGR")
    st_focused.markdown(f"### 상태: **{status_text}**")
    st_timer.metric("누적 집중 점수", f"{focus_score}")
    st_chart.line_chart(st.session_state.score_log[-100:])
    if st.session_state.warning_log:
        st_log.table(pd.DataFrame(st.session_state.warning_log, columns=["시각", "경고 내용"]))

cap.release()
st.success("세션 종료! 로그를 저장하거나 분석을 시작하세요.")

if st.button("💾 로그 저장하기", key="save_button"):
    log_df = pd.DataFrame(st.session_state.warning_log, columns=["시각", "경고 내용"])
    st.download_button("📥 다운로드", data=log_df.to_csv(index=False), file_name="focus_warning_log.csv")