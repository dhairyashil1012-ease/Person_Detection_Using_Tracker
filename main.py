import cv2
from ultralytics import YOLO
import os 
from process.sort_tracker import Sort , iou_batch
import numpy as np
from process.process_person import preprocess_p 


model = YOLO("/home/easemyai/sort_tracker_person_detection/models/yolo26n.pt") 


tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.2)


def inference(frame):
    if frame is None:
        print("Error: No image tensor provided for inference.")
        return frame # Return frame instead of None to prevent breaking video loops

    results = model.predict(frame, verbose=False) # verbose=False cleans up console logs
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    detection = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        detection.append([x1, y1, x2, y2, score])


    detection = np.array(detection)
        
    # 3. Update the global tracker instance
    track_Obj = tracker.update(detection)
    
    for tracking_output in track_Obj:
        x1, y1, x2, y2, track_id = tracking_output
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {track_id}"
        cv2.putText(frame,label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame



def main():
    video_path = '/home/easemyai/sort_tracker_person_detection/Inputs/8379244-uhd_3840_2160_25fps.mp4'
    preprocessed_stream=preprocess_p(video_path)
    # inference_stream = inference(preprocessed_stream)
    
    for annotated_frame in preprocessed_stream:
        # frame = cv2.resize(annotated_frame, (640, 640))

        inference_stream = inference(annotated_frame)
        # frame = cv2.resize(inference_stream, (640, 640))
        cv2.imshow("YOLO Tracking",inference_stream)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()











