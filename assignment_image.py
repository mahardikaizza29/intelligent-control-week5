from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Load model YOLOv8 Pose
pose_model = YOLO("yolov8n-pose.pt")

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Warna dan ketebalan garis
POSE_COLOR = (0, 255, 0)
HAND_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 255, 255)
THICKNESS = 2

# Nama-nama keypoint tubuh berdasarkan COCO
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections (COCO format)
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),   # Lengan kiri
    (6, 8), (8, 10),  # Lengan kanan
    (5, 6),           # Bahu
    (11, 13), (13, 15),  # Kaki kiri
    (12, 14), (14, 16),  # Kaki kanan
    (11, 12),          # Pinggul
    (5, 11), (6, 12)   # Bahu ke pinggul
]

# Baca gambar
image_path = "contoh.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Gambar tidak ditemukan atau format tidak didukung.")
    exit()

# Lakukan deteksi pose dengan YOLO
results = pose_model(frame)

# Loop untuk menggambar keypoints dan skeleton
for result in results:
    keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else None

    if keypoints is not None:
        for person_keypoints in keypoints:
            # Gambar keypoints tubuh + nama sendi
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f'{i} {KEYPOINT_NAMES[i]}', (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

            # Gambar skeleton tubuh
            for part_a, part_b in SKELETON_CONNECTIONS:
                if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                    x1, y1 = person_keypoints[part_a]
                    x2, y2 = person_keypoints[part_b]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), POSE_COLOR, THICKNESS)

# Konversi ke RGB untuk MediaPipe
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
hand_results = hands.process(frame_rgb)

# Gambar keypoints tangan dengan label angka
if hand_results.multi_hand_landmarks:
    for hand_landmarks in hand_results.multi_hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks.landmark):
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, HAND_COLOR, -1)
            cv2.putText(frame, f'{idx}', (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        # Gambar koneksi tangan
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

# Simpan dan tampilkan hasil deteksi
cv2.imwrite("pose_hand_result.jpg", frame)
cv2.imshow("Pose & Hand Detection with Labels", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

