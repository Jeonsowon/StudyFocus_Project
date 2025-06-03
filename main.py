import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import simpleaudio as sa

st.set_page_config(page_title="강의 집중도 측정", layout="wide")

# 상태 저장
if "score_log" not in st.session_state:
    st.session_state.score_log = []
if "score_timestamps" not in st.session_state:
    st.session_state.score_timestamps = []
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
if "last_score_update" not in st.session_state:
    st.session_state.last_score_update = time.time()
if "status_log" not in st.session_state:
    st.session_state.status_log = []

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
CALIBRATION_FRAMES = 60
EAR_THRESHOLD = 0.2
WARNING_DURATION = 10
STATUS_DURATION = 3
HEAD_MOVE_THRESHOLD_X = 60
HEAD_MOVE_THRESHOLD_Y = 40

head_off_start = None
eyes_closed_start = None
head_moved_warned = False
eyes_closed_warned = False
head_moved_status_start = None
eyes_closed_status_start = None

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
status_text = "Detecting..."

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

while cap.isOpened() and not stop:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    now = time.time()
    elapsed_time = now - st.session_state.start_time
    head_moved = False
    eyes_closed = False
    current_status = "✅ Focused"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose = landmarks[1]
            nx, ny = nose.x * w, nose.y * h

            if st.session_state.calibrated_center is None and st.session_state.frame_count < CALIBRATION_FRAMES:
                st.session_state.calibration_sum[0] += nx
                st.session_state.calibration_sum[1] += ny
                st.session_state.frame_count += 1
                current_status = f"기준점 설정 중 ({st.session_state.frame_count}/{CALIBRATION_FRAMES})"
                if st.session_state.frame_count == CALIBRATION_FRAMES:
                    cx = st.session_state.calibration_sum[0] / CALIBRATION_FRAMES
                    cy = st.session_state.calibration_sum[1] / CALIBRATION_FRAMES
                    st.session_state.calibrated_center = (cx, cy)
                continue

            if st.session_state.calibrated_center:
                cx, cy = st.session_state.calibrated_center
                dx = abs(cx - nx)
                dy = abs(cy - ny)
                if dx > HEAD_MOVE_THRESHOLD_X or dy > HEAD_MOVE_THRESHOLD_Y:
                    if not head_off_start:
                        head_off_start = now
                    if not head_moved_status_start:
                        head_moved_status_start = now
                    head_moved = True
                    if now - head_off_start >= WARNING_DURATION and not head_moved_warned:
                        play_alert()
                        hms = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                        st.session_state.warning_log.append((hms, "고개 이탈 경고"))
                        head_moved_warned = True
                else:
                    head_off_start = None
                    head_moved_warned = False
                    head_moved_status_start = None

            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < EAR_THRESHOLD:
                if not eyes_closed_start:
                    eyes_closed_start = now
                if not eyes_closed_status_start:
                    eyes_closed_status_start = now
                eyes_closed = True
                if now - eyes_closed_start >= WARNING_DURATION and not eyes_closed_warned:
                    play_alert()
                    hms = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
                    st.session_state.warning_log.append((hms, "눈 감김 경고"))
                    eyes_closed_warned = True
            else:
                eyes_closed_start = None
                eyes_closed_warned = False
                eyes_closed_status_start = None

    if eyes_closed_status_start and now - eyes_closed_status_start >= STATUS_DURATION:
        current_status = "😴 Eyes Closed"
    elif head_moved_status_start and now - head_moved_status_start >= STATUS_DURATION:
        current_status = "🧠 Head Moved"
    elif st.session_state.frame_count >= CALIBRATION_FRAMES:
        current_status = "✅ Focused"

    st_focused.markdown(f"### 상태: **{current_status}**")
    st.session_state.status_log.append((now, current_status))

    if now - st.session_state.last_score_update >= 60:
        window_start = st.session_state.last_score_update
        window_end = now
        focused_time = 0.0
        not_focused_time = 0.0
        previous_time = None
        previous_status = None

        for t, s in st.session_state.status_log:
            if t < window_start:
                continue
            if t > window_end:
                break
            if previous_time is not None:
                duration = t - previous_time
                if previous_status == "✅ Focused":
                    focused_time += duration
                else:
                    not_focused_time += duration
            previous_time = t
            previous_status = s

        if previous_time and previous_status:
            duration = now - previous_time
            if previous_status == "✅ Focused":
                focused_time += duration
            else:
                not_focused_time += duration

        if not_focused_time < 10:
            focus_score += 1

        minutes_passed = int((now - st.session_state.start_time) // 60)
        st.session_state.score_log.append(focus_score)
        st.session_state.score_timestamps.append(f"{minutes_passed}분")
        st.session_state.last_score_update = now

    FRAME_WINDOW.image(img, channels="BGR")
    st_timer.metric("누적 집중 점수", f"{focus_score}")
    if st.session_state.score_log:
        score_df = pd.DataFrame({"시간 (분)": st.session_state.score_timestamps, "점수": st.session_state.score_log})
        st_chart.line_chart(score_df.set_index("시간 (분)"))
    if st.session_state.warning_log:
        st_log.table(pd.DataFrame(st.session_state.warning_log, columns=["경과 시간", "경고 내용"]))

cap.release()

st.success("세션 종료! 아래에서 집중 점수 및 로그를 확인할 수 있습니다.")

center_col = st.columns([1, 2, 1])[1]

with center_col:
    st.subheader("📈 집중 점수 추이")
    if st.session_state.score_log:
        score_df = pd.DataFrame({"시간 (분)": st.session_state.score_timestamps, "점수": st.session_state.score_log})
        st.line_chart(score_df.set_index("시간 (분)"))

    if st.session_state.warning_log:
        st.subheader("⚠️ 경고 로그")
        st.table(pd.DataFrame(st.session_state.warning_log, columns=["경과 시간", "경고 내용"]))