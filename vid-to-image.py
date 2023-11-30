import cv2

video_path = 'videos/IMG_0054.MP4'
output_folder = 'frames/'

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    frame_filename = f"{output_folder}frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_filename, frame)

cap.release()