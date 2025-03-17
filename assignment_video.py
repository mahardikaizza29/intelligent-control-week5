from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import threading

# Load model YOLOv8 untuk mendeteksi pose dan objek manusia
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Warna dan ketebalan garis
WARNA = (0, 255, 0)
KETEBALAN = 2

# Inisialisasi titik-titik utama tubuh
TITIK_TUBUH = {
    "hidung": 0,
    "mata_kiri": 1, "mata_kanan": 2,
    "telinga_kiri": 3, "telinga_kanan": 4,
    "bahu_kiri": 5, "bahu_kanan": 6,
    "siku_kiri": 7, "siku_kanan": 8,
    "pergelangan_kiri": 9, "pergelangan_kanan": 10,
    "pinggul_kiri": 11, "pinggul_kanan": 12,
    "lutut_kiri": 13, "lutut_kanan": 14,
    "pergelangan_kaki_kiri": 15, "pergelangan_kaki_kanan": 16
}

# Pasangan titik (skeleton) berdasarkan format COCO
KONEKSI_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
    (1, 14), (14, 16)
]

# Nama titik berdasarkan format COCO
NAMA_TITIK = [
    "hidung", "mata_kiri", "mata_kanan", "telinga_kiri", "telinga_kanan",
    "bahu_kiri", "bahu_kanan", "siku_kiri", "siku_kanan",
    "pergelangan_kiri", "pergelangan_kanan", "pinggul_kiri", "pinggul_kanan",
    "lutut_kiri", "lutut_kanan", "pergelangan_kaki_kiri", "pergelangan_kaki_kanan"
]

# Buka video input dan buat video output
input_video_path = "input.mp4"
output_video_path = "output.mp4"
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# === Fungsi untuk deteksi objek (manusia) ===
def deteksi_objek(frame):
    results = object_model(frame)
    for result in results:
        kotak = result.boxes.xyxy.cpu().numpy()
        id_kelas = result.boxes.cls.cpu().numpy()
        kepercayaan = result.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(kotak, id_kelas, kepercayaan):
            if int(class_id) == 0:  # Manusia
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Manusia {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# === Fungsi untuk deteksi pose (kerangka) ===
def deteksi_pose(frame):
    results = pose_model(frame)
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        if keypoints is not None:
            for titik_manusia in keypoints:
                for i, (x, y) in enumerate(titik_manusia):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                for pasangan in KONEKSI_SKELETON:
                    bagian_a, bagian_b = pasangan
                    if bagian_a < len(titik_manusia) and bagian_b < len(titik_manusia):
                        x1, y1 = titik_manusia[bagian_a]
                        x2, y2 = titik_manusia[bagian_b]
                        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), WARNA, KETEBALAN)

# === Fungsi untuk deteksi tangan ===
def deteksi_tangan(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil_tangan = hands.process(frame_rgb)
    if hasil_tangan.multi_hand_landmarks:
        for landmark_tangan in hasil_tangan.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmark_tangan, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

# === Loop untuk membaca video ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    t1 = threading.Thread(target=deteksi_objek, args=(frame,))
    t2 = threading.Thread(target=deteksi_pose, args=(frame,))
    t3 = threading.Thread(target=deteksi_tangan, args=(frame,))

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
    
    out.write(frame)
    
    cv2.imshow("YOLOv8 + MediaPipe (Pemrosesan Video)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

