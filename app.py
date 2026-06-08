import os
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace
from scipy.spatial.distance import cosine

app = Flask(__name__)

# 1. Cấu hình đường dẫn hệ thống
DB_PATH = r"E:\xử lý ảnh số\khuong_mat\face_db.pkl"
THRESHOLD = 0.40
FRAME_SKIP = 3

# 2. Nạp cơ sở dữ liệu vector khuôn mặt từ ổ C
print("Đang tải cơ sở dữ liệu khuôn mặt...")
if os.path.exists(DB_PATH):
    with open(DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    print("Nạp dữ liệu thành công!")
else:
    face_db = {}
    print("CẢNH BÁO: Chưa tìm thấy file face_db.pkl. Vui lòng kiểm tra lại đường dẫn.")

# Khởi tạo bộ quét khuôn mặt nhanh Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    """Hàm tạo luồng camera liên tục và xử lý nhận diện hình ảnh"""
    cap = cv2.VideoCapture(0)
    frame_count = 0
    saved_predictions = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1) # Lật ảnh tạo hiệu ứng soi gương
        frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        # Chỉ xử lý trích xuất vector sau một số khung hình nhất định để tăng tốc độ mượt mà
        if frame_count % FRAME_SKIP == 0 or not saved_predictions:
            saved_predictions = []
            
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size == 0:
                    continue
                    
                name = "Unknown"
                min_dist = float("inf")
                
                try:
                    # Trích xuất vector từ khuôn mặt thời gian thực
                    embedding_objs = DeepFace.represent(
                        img_path=face_crop,
                        model_name="Facenet512",
                        enforce_detection=False
                    )
                    current_embedding = embedding_objs[0]["embedding"]
                    
                    # Đối chiếu khoảng cách toán học với kho dữ liệu đã gắn nhãn
                    for person_name, embeddings in face_db.items():
                        for db_embedding in embeddings:
                            dist = cosine(current_embedding, db_embedding)
                            if dist < min_dist:
                                min_dist = dist
                                if dist < THRESHOLD:
                                    name = person_name
                except Exception:
                    pass
                    
                saved_predictions.append((x, y, w, h, name, min_dist))
        
        # Vẽ các khung định danh lên ảnh
        for (x, y, w, h, name, min_dist) in saved_predictions:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            text = f"{name} ({min_dist:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Mã hóa khung hình thành định dạng JPEG để truyền tải qua môi trường Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Trả về luồng dữ liệu kiểu Multipart định dạng chuẩn video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """Route chính hiển thị giao diện người dùng"""
    return render_template('index.html', people=list(face_db.keys()))

@app.route('/video_feed')
def video_feed():
    """Route cung cấp luồng dữ liệu video cho thẻ <img> trên HTML"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """API endpoint để nhận các thông số điều chỉnh từ giao diện Web"""
    global THRESHOLD, FRAME_SKIP
    data = request.get_json()
    if 'threshold' in data:
        THRESHOLD = float(data['threshold'])
    if 'frame_skip' in data:
        FRAME_SKIP = int(data['frame_skip'])
    return jsonify({"status": "success", "threshold": THRESHOLD, "frame_skip": FRAME_SKIP})

if __name__ == '__main__':
    # Khởi chạy server Flask tại cổng mặc định 5000
    app.run(host='0.0.0.0', port=5000, debug=False)