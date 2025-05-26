import cv2
import mediapipe as mp
import numpy as np
import time
import simpleaudio as sa

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 경고음 재생 함수
# def play_alert():
#     wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
#     wave_obj.play()

# 눈 EAR 계산 함수
def calculate_ear(landmarks, eye_indices, w, h):
    p1 = np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h])
    p2 = np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h])
    p3 = np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h])
    p4 = np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h])
    p5 = np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h])
    p6 = np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h])
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# 눈 인덱스 (Mediapipe 기준)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# 파라미터
CALIBRATION_FRAMES = 60
EAR_THRESHOLD = 0.2
CLOSE_DURATION = 10  # 눈 감김 경고 시간
HEAD_DURATION = 10  # 머리 비정렬 경고 시간

# 추적 변수
frame_count = 0
sum_x, sum_y = 0, 0
center_x, center_y = None, None
not_focused_start = None
eyes_closed_start = None

# 실행
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose = landmarks[1]
            x = nose.x * w
            y = nose.y * h

            # 기준 좌표 캘리브레이션
            if center_x is None and frame_count < CALIBRATION_FRAMES:
                sum_x += x
                sum_y += y
                frame_count += 1
                cv2.putText(img, f"Calibrating... ({frame_count}/{CALIBRATION_FRAMES})",
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
                if frame_count == CALIBRATION_FRAMES:
                    center_x = sum_x / CALIBRATION_FRAMES
                    center_y = sum_y / CALIBRATION_FRAMES
                    print(f"[기준 설정 완료] 기준좌표: ({center_x:.1f}, {center_y:.1f})")
                continue

            # 코 위치 기반 집중 판단
            dx = abs(center_x - x)
            dy = abs(center_y - y)
            head_focused = dx < 80 and dy < 60

            # 눈 감김 판단
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            eyes_open = avg_ear >= EAR_THRESHOLD

            # 눈 감김 지속 추적
            if not eyes_open:
                if not eyes_closed_start:
                    eyes_closed_start = time.time()
                else:
                    eye_elapsed = time.time() - eyes_closed_start
                    cv2.putText(img, f"Eyes Closed ({int(eye_elapsed)}s)", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if eye_elapsed >= CLOSE_DURATION:
                        play_alert()
                        eyes_closed_start = None
            else:
                eyes_closed_start = None

            # 머리 비정렬 지속 추적
            if head_focused:
                not_focused_start = None
                cv2.putText(img, "Head Aligned", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                if not not_focused_start:
                    not_focused_start = time.time()
                else:
                    head_elapsed = time.time() - not_focused_start
                    cv2.putText(img, f"Head Off ({int(head_elapsed)}s)", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if head_elapsed >= HEAD_DURATION:
                        play_alert()
                        not_focused_start = None

            # 디버깅 표시
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 0), -1)
            cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 255, 0), 2)

    cv2.imshow("Focus Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()