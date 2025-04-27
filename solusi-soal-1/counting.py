import cv2
import numpy as np

class LineCrossingCounter:
    
    def __init__(self, line_start, line_end):
        self.line = (line_start[0], line_start[1], line_end[0], line_end[1])
        
        self.a = self.line[3] - self.line[1]
        self.b = self.line[0] - self.line[2]
        self.c = self.line[2]*self.line[1] - self.line[0]*self.line[3]  # x2*y1 - x1*y2
        
        self.prev_centers = {}
        self.prev_positions = {}
        
        self.class_counts = {}
        
        self.counted_objects = set()
        
        self.trajectory_history = {}
        self.history_length = 10
        
    def update(self, frame, track_ids, boxes, class_names):
        crossed_objects = []
        
        for i, (track_id, box, class_name) in enumerate(zip(track_ids, boxes, class_names)):
            track_id = int(track_id)
            
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            current_center = (center_x, center_y)
            
            if class_name not in self.class_counts:
                self.class_counts[class_name] = 0
            
            if track_id not in self.trajectory_history:
                self.trajectory_history[track_id] = []
            
            self.trajectory_history[track_id].append(current_center)
            
            if len(self.trajectory_history[track_id]) > self.history_length:
                self.trajectory_history[track_id].pop(0)
            
            if len(self.trajectory_history[track_id]) >= 2 and track_id not in self.counted_objects:
                prev_center = self.trajectory_history[track_id][-2]
                
                prev_sign = np.sign(self.a * prev_center[0] + self.b * prev_center[1] + self.c)
                curr_sign = np.sign(self.a * current_center[0] + self.b * current_center[1] + self.c)
                
                if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:
                    self.class_counts[class_name] += 1
                    
                    self.counted_objects.add(track_id)
                    
                    t = -(self.a * prev_center[0] + self.b * prev_center[1] + self.c) / (
                        self.a * (current_center[0] - prev_center[0]) + 
                        self.b * (current_center[1] - prev_center[1])
                    )
                    
                    t = max(0, min(1, t))
                    
                    cross_x = int(prev_center[0] + t * (current_center[0] - prev_center[0]))
                    cross_y = int(prev_center[1] + t * (current_center[1] - prev_center[1]))
                    
                    crossed_objects.append({
                        'track_id': track_id,
                        'class_name': class_name,
                        'cross_point': (cross_x, cross_y)
                    })
                    
                    cv2.circle(frame, (cross_x, cross_y), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{class_name}", (cross_x + 10, cross_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            self.prev_centers[track_id] = current_center
            
        return crossed_objects
    
    def draw_line(self, frame):
        cv2.line(frame, (self.line[0], self.line[1]), (self.line[2], self.line[3]), (0, 255, 0), 2)
        
        cv2.circle(frame, (self.line[0], self.line[1]), 5, (255, 0, 0), -1)  # Start point
        cv2.circle(frame, (self.line[2], self.line[3]), 5, (0, 0, 255), -1)  # End point
        
        midpoint_x = (self.line[0] + self.line[2]) // 2
        midpoint_y = (self.line[1] + self.line[3]) // 2
        cv2.putText(frame, "Counting Line", (midpoint_x - 50, midpoint_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_counts(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, "Object Counts:", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 60
        for i, (class_name, count) in enumerate(self.class_counts.items()):
            if class_name == "Ripe":
                color = (0, 255, 0)  # Green
            elif class_name == "Unripe":
                color = (255, 255, 0)  # Yellow
            elif class_name == "OverRipe":
                color = (0, 165, 255)  # Orange
            elif class_name == "Rotten":
                color = (0, 0, 255)  # Red
            elif class_name == "EmptyBunch":
                color = (128, 128, 128)  # Gray
            else:
                color = (255, 255, 255)  # White
                
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (15, y_offset + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def get_counts(self):
        return self.class_counts