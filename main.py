from google.colab import files
import cv2, math
from ultralytics import YOLO

model_upload = files.upload()
model_path = list(model_upload.keys())[0]

video_upload = files.upload()
video_path = list(video_upload.keys())[0]

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output_stable.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

tracks = []
ALPHA = 0.3
MAX_DIST = 60

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            if bw < 30 or bh < 6:
                continue
            if bw / bh < 3:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((cx, cy, x1, y1, x2, y2))

    new_tracks = []
    used = set()

    for t in tracks:
        best_i, best_d = -1, 1e9
        for i, (cx, cy, *_ ) in enumerate(detections):
            if i in used:
                continue
            d = dist((t['cx'], t['cy']), (cx, cy))
            if d < best_d:
                best_d, best_i = d, i

        if best_i != -1 and best_d < MAX_DIST:
            cx, cy, x1, y1, x2, y2 = detections[best_i]
            used.add(best_i)
            ema_x = int(ALPHA * cx + (1 - ALPHA) * t['ema_x'])
            ema_y = int(ALPHA * cy + (1 - ALPHA) * t['ema_y'])
            new_tracks.append({
                'cx': cx, 'cy': cy, 'ema_x': ema_x, 'ema_y': ema_y, 'box': (x1, y1, x2, y2)
            })

    for i, (cx, cy, x1, y1, x2, y2) in enumerate(detections):
        if i in used:
            continue
        new_tracks.append({
            'cx': cx, 'cy': cy, 'ema_x': cx, 'ema_y': cy, 'box': (x1, y1, x2, y2)
        })

    tracks = new_tracks

    for t in tracks:
        x1, y1, x2, y2 = t['box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        bar_h = y2 - y1
        tx = t['ema_x']
        ty = t['ema_y'] + int(bar_h * 2.5)
        cv2.drawMarker(frame, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    out.write(frame)

cap.release()
out.release()
files.download("output_stable.mp4")