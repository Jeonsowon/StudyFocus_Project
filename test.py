# main.py
import cv2
import mediapipe as mp
import time
import simpleaudio as sa

# 얼굴 추적 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# 얼굴 방향 판단 기준 (yaw, pitch 대체)
def is_focused(landmarks, img_w, img_h):
    nose_tip = landmarks[1]  # 코 끝 좌표
    x = nose_tip.x * img_w
    y = nose_tip.y * img_h
    cx = img_w / 2
    cy = img_h / 2
    dx = abs(cx - x)
    dy = abs(cy - y)
    return dx < 80 and dy < 60  # 중심으로부터 얼마나 벗어났는지 기준

# 경고음
def play_alert():
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    wave_obj.play()

# 실행
cap = cv2.VideoCapture(0)
not_focused_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    focused = True
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            focused = is_focused(landmarks.landmark, w, h)
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # 집중 상태 확인
    if focused:
        not_focused_start = None
        cv2.putText(frame, "Focused", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        if not not_focused_start:
            not_focused_start = time.time()
        else:
            duration = time.time() - not_focused_start
            cv2.putText(frame, f"Not Focused ({int(duration)}s)", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if duration >= 10:
                play_alert()
                not_focused_start = None  # 리셋

    cv2.imshow("Focus Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()