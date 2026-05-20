import cv2

def preprocess_p(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

        yield resized_frame
        
    cap.release()

# p=preprocess_p('/home/easemyai/sort_tracker_person_detection/Inputs/987-1280x720-1-result.mp4')
# print(p)