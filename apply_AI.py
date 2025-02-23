import cv2
import numpy as np
import time
import requests
import threading
import queue
from ultralytics import YOLO
from scipy.spatial import distance

def send_coordinates_to_raspberry(coordinates):
    url = 'http://192.168.244.146:5000/receive_coordinates'
    payload = {'coordinates': coordinates}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Coordinates sent successfully!")
        else:
            print("Failed to send coordinates.")
    except requests.exceptions.RequestException as e:
        print("Error sending data:", e)

# Load YOLO segmentation model
model = YOLO(r'D:\download\best.pt')

# Global variables
prev_segments = []
selected_segment_id = None  # Lưu ID của phân đoạn được chọn
last_update_time = 0  # Thời gian cuối cùng gửi tọa độ
coordinates_queue = queue.Queue() 

def approximate_polygon(polygon, epsilon_ratio=0.01): 
    contour = np.array(polygon, dtype=np.int32) 
    epsilon = epsilon_ratio * cv2.arcLength(contour, True) 
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True) 
    return [tuple(int(coord) for coord in point[0]) for point in approx_polygon]

def click_event(event, x, y, flags, param):
    """
    Xử lý sự kiện click chuột để chọn ô đỗ.
    """
    global selected_segment_id
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_segment_id = None  # Reset lựa chọn
        min_distance = float('inf')  
        for segment in param:  # Duyệt qua tất cả các phân đoạn
            segment_coords = segment['coords']
            if cv2.pointPolygonTest(np.array(segment_coords, dtype=np.int32), (x, y), False) >= 0:
                center = calculate_center(segment_coords)
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)

                if distance < min_distance:
                    min_distance = distance
                    selected_segment_id = segment['id']
                
                
        if selected_segment_id is not None:
            print(f"Selected segment:")
        else:
            print("No segment selected.")

def draw_segments(image, results):
    """
    Vẽ phân đoạn lên ảnh và cập nhật tọa độ ô đỗ được chọn.
    """
    global selected_segment_id, last_update_time, prev_segments
    segment_data = []  # Lưu danh sách phân đoạn hiện tại
    selected_segment_found = False

    for result in results:
        for idx, box in enumerate(result.boxes.xyxy):  # Bounding box (x1, y1, x2, y2)
            cls = int(result.boxes.cls[idx])  # Class ID
            #confidence = float(result.boxes.conf[idx])  # Confidence score
            label = f"{model.names[cls]}"

            # Lấy tọa độ phân đoạn
            if cls in [0, 1]:  # Chỉ xử lý ID 0 và 1
                if result.masks and idx < len(result.masks.xy):
                    segment_coords = [(int(pt[0]), int(pt[1])) for pt in result.masks.xy[idx]]
                    segment_coords = approximate_polygon(segment_coords)

                    # Kiểm tra nếu là phân đoạn được chọn
                    if selected_segment_id == idx:
                        color = (255, 255, 0)  # Màu vàng cho phân đoạn được chọn
                        selected_segment_found = True 
                    elif cls == 0:
                        color = (0, 255, 0)  # Màu xanh lá
                    elif cls == 1:
                        color = (0, 0, 255)  # Màu đỏ

                    # Vẽ phân đoạn
                    cv2.polylines(image, [np.array(segment_coords, dtype=np.int32)], isClosed=True, color=color, thickness=2)
                    cv2.putText(image, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Lưu phân đoạn
                    segment_data.append({'id': idx, 'coords': segment_coords, 'center': calculate_center(segment_coords)})

    if selected_segment_id is not None and not selected_segment_found:
        print(f"Selected segment has disappeared.")
        selected_segment_id = None  # Hủy chọn phân đoạn nếu nó không còn tồn tại

    # Cập nhật tọa độ ô đỗ được chọn
    if selected_segment_id is not None:
        current_time = time.time()
        for segment in segment_data:
            if segment['id'] == selected_segment_id:
                # Hiển thị tọa độ mới nhất
                if current_time - last_update_time >= 3:  # Gửi mỗi 3 giây
                    last_update_time = current_time
                    data_to_send = {
                        'points': segment['coords']
                    }
                    print(f"Sending data: {data_to_send}")
                    coordinates_queue.put(data_to_send)  # Đưa dữ liệu vào hàng đợi

                break
    

    prev_segments = segment_data

    return image, segment_data

def calculate_center(coords):
    """
    Tính toán tọa độ trung tâm của vùng.
    """
    x_coords = [pt[0] for pt in coords]
    y_coords = [pt[1] for pt in coords]
    return sum(x_coords) // len(coords), sum(y_coords) // len(coords)


def find_closest_segment(current_segments, previous_segments, max_distance=50):
    """
    Tìm phân đoạn gần nhất với phân đoạn đã chọn trong khung hình trước.
    """
    if not current_segments or not previous_segments:
        return None

    closest_segment = None
    min_distance = float('inf')
    for prev_seg in previous_segments:
        for curr_seg in current_segments:
            dist = distance.euclidean(curr_seg['center'], prev_seg['center'])
            if dist < min_distance and dist < max_distance:
                min_distance = dist
                closest_segment = curr_seg['id']

    return closest_segment

def send_coordinates():
    """
    Luồng xử lý gửi tọa độ sang Raspberry Pi.
    """
    while True:
        try:
            data_to_send = coordinates_queue.get(timeout=1)  # Chờ dữ liệu trong hàng đợi
            #print(f"Sending data: {data_to_send}")
            send_coordinates_to_raspberry(data_to_send)  # Hàm gửi dữ liệu
            coordinates_queue.task_done()  # Đánh dấu nhiệm vụ đã hoàn thành
        except queue.Empty:
            pass  # Nếu không có dữ liệu, tiếp tục vòng lặp


# Main loop
if __name__ == "__main__":
    # Khởi động luồng gửi tọa độ
    send_thread = threading.Thread(target=send_coordinates, daemon=True)
    send_thread.start()
    # Initialize video capture for cameras
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(4)
    cap4 = cv2.VideoCapture(3)

    background = np.full((400, 400, 3), 128, dtype=np.uint8)  # Background for combined video

    # Tọa độ điểm nguồn của video camera (giả sử video có kích thước 350x800)
    pts_src1 = np.array([[0, 0], [0, 400], [173, 400], [173, 0]], dtype='float32')
    pts_src2 = np.array([[0, 0], [140, 0], [140, 141], [0, 141]], dtype='float32')
    pts_src3 = np.array([[0, 0], [140, 0], [140, 141], [0, 141]], dtype='float32')
    pts_src4 = np.array([[0, 0], [0, 400], [173, 400], [173, 0]], dtype='float32')

    # Tọa độ điểm đích trong nền 800x800
    pts_dst1 = np.array([[0, 0], [173, 0], [173, 400], [0, 400]], dtype='float32')
    pts_dst2 = np.array([[293, 0], [107, 0], [107, 141], [293, 141]], dtype='float32')
    pts_dst3 = np.array([[107, 400], [293, 400], [293, 258], [107, 258]], dtype='float32')
    pts_dst4 = np.array([[400, 400], [227, 400], [227, 0], [400, 0]], dtype='float32')

    # Transformation matrices for perspective warping
    h1, _ = cv2.findHomography(pts_src1, pts_dst1)
    h2, _ = cv2.findHomography(pts_src2, pts_dst2)
    h3, _ = cv2.findHomography(pts_src3, pts_dst3)
    h4, _ = cv2.findHomography(pts_src4, pts_dst4)

    cv2.namedWindow("Parking Detection")


    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            print("Error: Could not read frame.")
            break

        # Resize và warp các khung hình
        resized_frame1 = cv2.resize(frame1, (175, 400))
        resized_frame2 = cv2.resize(frame2, (161, 205))
        resized_frame3 = cv2.resize(frame3, (160, 205))
        resized_frame4 = cv2.resize(frame4, (175, 400))

        resized_frame1 = cv2.flip(resized_frame1, 1) # 0: lật dọc, 1: lật ngang, -1: lật cả ngang và dọc
        resized_frame2 = cv2.flip(resized_frame2, 1) 
        resized_frame3 = cv2.flip(resized_frame3, 1) 
        resized_frame4 = cv2.flip(resized_frame4, 1) 

        warped_frame1 = cv2.warpPerspective(resized_frame1, h1, (400, 400))
        warped_frame2 = cv2.warpPerspective(resized_frame2, h2, (400, 400))
        warped_frame3 = cv2.warpPerspective(resized_frame3, h3, (400, 400))
        warped_frame4 = cv2.warpPerspective(resized_frame4, h4, (400, 400))

        # Tạo mặt nạ với kích thước và kênh giống nền
        mask1 = np.zeros_like(background, dtype=np.uint8)
        cv2.fillConvexPoly(mask1, np.array([[0, 0], [0, 400], [107, 400], [173, 258], [173, 141], [107, 0]], dtype=np.int32), (255, 255, 255))

        mask2 = np.zeros_like(background, dtype=np.uint8)
        cv2.fillConvexPoly(mask2, np.array([[293, 0], [107, 0], [173, 141], [227, 141]], dtype=np.int32), (255, 255, 255))

        mask3 = np.zeros_like(background, dtype=np.uint8)
        cv2.fillConvexPoly(mask3, np.array([[107, 400], [293, 400], [227, 258], [173, 258]], dtype=np.int32), (255, 255, 255))

        mask4 = np.zeros_like(background, dtype=np.uint8)
        #cv2.fillConvexPoly(mask4, np.array([[587, 0], [455, 282], [455, 517], [587, 800], [800, 800], [800, 0]], dtype=np.int32), (255, 255, 255))
        cv2.fillConvexPoly(mask4, np.array([[293, 0], [400, 0], [400, 400], [293, 400], [227, 258], [227, 141]], dtype=np.int32), (255, 255, 255))


        # Áp dụng mask và kết hợp các khung hình
        masked_background1 = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)))
        result1 = cv2.add(masked_background1, cv2.bitwise_and(warped_frame1, warped_frame1, mask=cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)))

        masked_background2 = cv2.bitwise_and(result1, result1, mask=cv2.bitwise_not(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)))
        result2 = cv2.add(masked_background2, cv2.bitwise_and(warped_frame2, warped_frame2, mask=cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)))

        masked_background3 = cv2.bitwise_and(result2, result2, mask=cv2.bitwise_not(cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)))
        result3 = cv2.add(masked_background3, cv2.bitwise_and(warped_frame3, warped_frame3, mask=cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)))

        masked_background4 = cv2.bitwise_and(result3, result3, mask=cv2.bitwise_not(cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)))
        result4 = cv2.add(masked_background4, cv2.bitwise_and(warped_frame4, warped_frame4, mask=cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)))

        # # Tổng hợp các khung hình
        # result = background.copy()
        # result = cv2.add(result, result1)
        # result = cv2.add(result, result2)
        # result = cv2.add(result, result3)
        # result = cv2.add(result, result4)


        # Apply YOLO model to detect parking spots
        results = model(result4, verbose=False)
        result4, segment_data = draw_segments(result4, results)
        #result4, segment_data = draw_segments(result4, results)

        # Highlight only the selected segment
        # if selected_segment is not None:
        #     cv2.polylines(result4, [np.array(selected_segment, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=3)

        # Set mouse callback for selecting parking spots
        cv2.setMouseCallback("Parking Detection", click_event, param=segment_data)

        # Display the result
        cv2.imshow("Parking Detection", result4)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()
    cv2.destroyAllWindows()
