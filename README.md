StudyFocus_Project
============================
Project Overview 🖥️
---------------------------
### 개발 환경
  > window10, python3.10
### 개발 목적
  > 학습 환경이 오프라인에서 온라인으로 변화되고 있습니다. 집에서 혼자 온라인으로 강의를 보는 것은 강의실에서 실시간으로 수업을 듣는 것보다 집중도가 떨어질 수 있습니다. 그래서 온라인으로 학습 중인 사용자의 집중 상태를 실시간으로 분석하고 피드백 줄 수 있는 도구를 생각했습니다.

  > 사용자의 눈 감김이나 고개 이탈 등 주의가 흐트러진 상태가 지속됨을 경고하고 집중 점수를 피드백하여 스스로 학습 상태를 점검할 수 있도록 하였습니다. 또한 집중력이 떨어진 순간을 기록하여 복습이 필요한 부분을 확인하여 학습 효율을 향상시킬 수 있는 도구로 개발하였습니다.
### 기술 스택
  >* Python 3.10 이상
  >* OpenCV: 실시간 비디오 처리
  >* MediaPipe: 얼굴 랜드마크 추적
  >* Streamlit: 웹 기반 실시간 UI 구성
  >* SimpleAudio: 경고음 재생
  >* Pandas: 로그 테이블
>  * NumPy: 수치 계산

-----------------------------
✨ Project Description ✨
-----------------------------
+ 파일 설명
  - main.py   
    : 프로그램이 진행되는 python source code.
  - alarm.wav   
    : 주의 흐트러진 상태로 감지하면 울리는 경고음
  - requirements.txt   
    : 설치 필요 라이브러리 명세서

+ 동작 흐름
1. 초기 60프레임 동안 코 위치 기준점 설정
2. 이후 실시간으로 눈 감김 및 고개 이탈 여부 감지
3. 눈 감김 또는 고개 이탈이 10초 이상 지속되면:  
  - 경고음 재생  
  - 로그 기록  
4. 1분 단위로 집중 상태 유지 시 점수 증가
5. 실시간 UI를 통해 상태 및 그래프 제공

+ UI 구성
  - 좌측: 실시간 웹캠 영상
  - 우측: 상태 텍스트, 점수 및 타이머, 실시간 그래프, 경고 로그 테이블
    
+ <main.py> 코드 설명
    ``` python
    import streamlit as st
    import cv2
    import mediapipe as mp
    import numpy as np
    import time
    import pandas as pd
    import simpleaudio as sa
    ```
    > pip install and import library 필요

    ```
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
    if "last_score_update" not in st.session_state:
        st.session_state.last_score_update = time.time()
    ```
    > Streamlit은 세션마다 상태가 초기화되므로 st.session_state를 통해 지속적으로 정보를 저장
    > score_log, warning_log, start_time 등은 점수 누적, 경고 기록, 타이머 등을 위해 필요
  
    ```
    # Mediapipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]
    ```
    > refine_landmarks=True는 눈과 입 등의 정밀한 위치까지 추적 가능
    > EAR 계산을 위해 필요한 눈 좌표 인덱스

    ```
    CALIBRATION_FRAMES = 60
    EAR_THRESHOLD = 0.2
    WARNING_DURATION = 10
    HEAD_MOVE_THRESHOLD_X = 60
    HEAD_MOVE_THRESHOLD_Y = 40
    ```
    > 고개 이탈의 기준점 설정 시간으로 60프레임
    > 눈 감김과 고개 이탈 인식 기준 설정
    > 비집중 상태 10초 지속되면 경고음 기준 설정

    ```
    head_off_start = None
    eyes_closed_start = None
    head_moved_warned = False
    eyes_closed_warned = False
    
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
    ```
    > 변수 초기 설정 및 웹캠 열기
    > Streamlit UI 구성으로 좌측에 실시간 웹캠 표시, 우측 상태 텍스트, 점수, 그래프, 경고 로그 표시

    ```
    def calculate_ear(landmarks, eye_indices, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    ear = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * np.linalg.norm(p[0] - p[3]))
    return ear
    ```
    >EAR (Eye Aspect Ratio): 눈의 세로 길이 대비 가로 길이를 비율로 나타낸 값을 계산하는 함수
    >값이 낮을수록 눈을 감은 상태로 간주
    
    ```
    def play_alert():
    try:
        wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
        wave_obj.play()
    except:
        pass
    ```
    > 사용자에게 경고음을 통해 스스로 집중 상태를 점검하도록 시각적인 피드백 외에도 청각적인 알림 설정 함수

    ```
    while cap.isOpened() and not stop:
    ```
    > 웹캡에서 실시간 프레임 받아 분석하는 메인 루프 시작
    
    ```
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
    ```
    > 초기 60프레임 동안 코의 위치 평균을 계산하여 기준점으로 설정
    
    ```
    if st.session_state.calibrated_center:
    cx, cy = st.session_state.calibrated_center
    dx = abs(cx - nx)
    dy = abs(cy - ny)
    if dx > HEAD_MOVE_THRESHOLD_X or dy > HEAD_MOVE_THRESHOLD_Y:
        if not head_off_start:
            head_off_start = now
        elif now - head_off_start >= WARNING_DURATION:
            head_moved = True
            if not head_moved_warned:
                play_alert()
                st.session_state.warning_log.append((time.strftime("%H:%M:%S"), "고개 이탈 경고"))
                head_moved_warned = True
    else:
        head_off_start = None
        head_moved_warned = False
    ```
    > 기준점 대비 X 또는 Y 방향으로 기준치 이상 벗어나고, 10초 이상 지속되면 고개 이탈 감지 및 경고음 발생.

    ```
    left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
    right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
    avg_ear = (left_ear + right_ear) / 2.0
    if avg_ear < EAR_THRESHOLD:
        if not eyes_closed_start:
            eyes_closed_start = now
        elif now - eyes_closed_start >= WARNING_DURATION:
            eyes_closed = True
            if not eyes_closed_warned:
                play_alert()
                st.session_state.warning_log.append((time.strftime("%H:%M:%S"), "눈 감김 경고"))
                eyes_closed_warned = True
    else:
        eyes_closed_start = None
        eyes_closed_warned = False
    ```
    > EAR 계산으로 눈 감김 여부를 판단하여 10초 이상 지속되면 눈 감김 감지 및 경고음 발생
    
    ```
    if eyes_closed:
        current_status = "😴 Eyes Closed"
    elif head_moved:
        current_status = "🧠 Head Moved"
    else :
        current_status = "✅ Focused"

    st_focused.markdown(f"### 상태: **{current_status}**")
    ```
    > 현재 상태를 실시간으로 텍스트로 반영하여 사용자에게 시각 피드백 제공

    ```
    if now - st.session_state.last_score_update >= 60:
    if not head_moved and not eyes_closed:
        focus_score += 1
    st.session_state.last_score_update = now
    st.session_state.score_log.append(focus_score)
    ```
    > 1분 단위로 집중이 유지될 경우 점수 1점 추가
    > 비집중 감지되면 점수 상승 없음.

    ```
    FRAME_WINDOW.image(img, channels="BGR")
    st_timer.metric("누적 집중 점수", f"{focus_score}")
    st_chart.line_chart(st.session_state.score_log[-100:])
    if st.session_state.warning_log:
        st_log.table(pd.DataFrame(st.session_state.warning_log, columns=["시각", "경고 내용"]))
    ```
    > 카메라 영상, 상태 텍스트, 점수, 그래프, 로그 테이블 모두 실시간 갱신

    ```
    if st.button("💾 로그 저장하기", key="save_button"):
      log_df = pd.DataFrame(st.session_state.warning_log, columns=["시각", "경고 내용"])
      st.download_button("📥 다운로드", data=log_df.to_csv(index=False), file_name="focus_warning_log.csv")
    ```
    > 경고 로그를 CSV 파일로 다운로드할 수 있어 사용자 기록 활용 가능.

