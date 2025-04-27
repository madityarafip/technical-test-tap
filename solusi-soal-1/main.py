from counting import LineCrossingCounter
from ultralytics import YOLO
import numpy as np
import argparse
import time
import math
import cv2
        
def yolo_inference_video_counting(model, video_path, counter, class_map):
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video Properties:")
    print(f"- Resolution: {width}x{height}")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {total_frames}")
    print(f"- Duration: {int(duration // 60):02d}:{int(duration % 60):02d}")

    output_path = "result/output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving output to: {output_path}")

    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_result = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)
            annotated_frame = results[0].plot()
            
            annotated_frame = counter.draw_line(annotated_frame)

            if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                class_names = [class_map[results[0].names[class_id]] for class_id in class_ids]
                
                crossed_objects = counter.update(annotated_frame, track_ids, boxes, class_names)
                
                for obj in crossed_objects:
                    cv2.circle(annotated_frame, obj['cross_point'], 5, (0, 0, 255), -1)
            
            # Draw counts on frame
            annotated_frame = counter.draw_counts(annotated_frame)

            cv2.imshow("YOLO11 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            output_writer.write(annotated_frame)

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("Process interrupted by user.")

def main():
    # Create parser
    parser = argparse.ArgumentParser(description="YOLO video inference arguments config")

    # Add arguments
    parser.add_argument("--model", type=str, help="YOLO model path")
    parser.add_argument("--input", type=str, help="Video input path")
    parser.add_argument("--xyxy", type=str, help="Your counting line coordinates ex: '[5,10,5,50]'")

    # Parse Arguments
    args = parser.parse_args()

    # Load Model
    model_path = args.model
    print(f"Load YOLO model from: {model_path}")
    model = YOLO(model_path)

    coords = args.xyxy.strip('[]')
    coords = [int(x.strip()) for x in coords.split(',')]
    line_start, line_end = (coords[0], coords[1]), (coords[2], coords[3])
    counter = LineCrossingCounter(line_start, line_end)


    class_map = {
              "0": "Ripe",
              "1": "Unripe",
              "2": "OverRipe",
              "3": "Rotten",
              "4": "EmptyBunch"
    }

    yolo_inference_video_counting(
        model,
        args.input,
        counter,
        class_map
    )

if __name__ == "__main__":
    main()