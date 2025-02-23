import cv2
import numpy as np

# Tạo nền xám với kích thước 800x800
background = np.full((400, 400, 3), 128, dtype=np.uint8)  # Nền xám


# Initialize video capture for cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(4)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(2)

#background = np.full((400, 400, 3), 128, dtype=np.uint8)  # Background for combined video

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

    cv2.imshow('Result3', result4)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cv2.destroyAllWindows()

