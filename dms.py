import argparse
import cv2
import numpy as np
import torch
import tensorflow as tf
import time
import pygame

from dms_utils.dms_utils import load_and_preprocess_image, ACTIONS
from net import MobileNet
from facial_tracking.facialTracking import FacialTracker
import facial_tracking.conf as conf

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.wav')  # Path to the alert sound file

# Initialize variables to track eye closure duration and yawning status
eye_closed_start_time = None
yawn_frame_count = 0
YAWN_FRAME_THRESHOLD = 5  # Threshold for consecutive yawning frames
EYE_CLOSED_DURATION_THRESHOLD = 2  # Threshold for eye closure duration in seconds

# Flicker parameters
flicker_intensity = 0
flicker_direction = 1
show_warning = False
frame_count = 0

def trigger_warning_signal():
    global show_warning
    show_warning = True
    print("Warning: Drowsiness Detected! Red light flickering...")
    alert_sound.play()  # Play alert sound

def create_warning_frame(frame, intensity, show_warning, frame_count):
    if show_warning:
        red_overlay = np.zeros(frame.shape, dtype=np.uint8)
        red_overlay[:] = (0, 0, 255)  # BGR color format
        frame = cv2.addWeighted(frame, 1, red_overlay, intensity, 0)
        
        # Add "WARNING!" text with flashing effect
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 2.5
        text_thickness = 5
        text_size = cv2.getTextSize("WARNING!", font, text_scale, text_thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        
        # Flashing text color (alternates between black and red)
        text_color = (0, 0, 0) if frame_count % 30 < 15 else (0, 0, 255)
        
        # Add a white outline to make text visible on dark background
        cv2.putText(frame, "WARNING!", (text_x, text_y), font, text_scale, (255, 255, 255), text_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, "WARNING!", (text_x, text_y), font, text_scale, text_color, text_thickness, cv2.LINE_AA)

        # Add "Alert: Vehicle Slowing Down" text
        alert_text_line1 = "Alert: Drowsiness Detected"
        alert_text_line2 = "Vehicle Slowing Down"
        alert_text_scale = 1.5
        alert_text_thickness = 3
        alert_text_size1 = cv2.getTextSize(alert_text_line1, font, alert_text_scale, alert_text_thickness)[0]
        alert_text_x1 = (frame.shape[1] - alert_text_size1[0]) // 2
        alert_text_y1 = text_y + 100

        alert_text_size2 = cv2.getTextSize(alert_text_line2, font, alert_text_scale, alert_text_thickness)[0]
        alert_text_x2 = (frame.shape[1] - alert_text_size2[0]) // 2
        alert_text_y2 = alert_text_y1 + alert_text_size1[1]+20
        
       # Flashing alert text color (same effect as WARNING!)
        alert_text_color = (0, 0, 0) if frame_count % 30 < 15 else (0, 0, 255)

        # Draw both lines of the alert text
        cv2.putText(frame, alert_text_line1, (alert_text_x1, alert_text_y1), font, alert_text_scale, (255, 255, 255), alert_text_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, alert_text_line1, (alert_text_x1, alert_text_y1), font, alert_text_scale, alert_text_color, alert_text_thickness, cv2.LINE_AA)

        cv2.putText(frame, alert_text_line2, (alert_text_x2, alert_text_y2), font, alert_text_scale, (255, 255, 255), alert_text_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, alert_text_line2, (alert_text_x2, alert_text_y2), font, alert_text_scale, alert_text_color, alert_text_thickness, cv2.LINE_AA)
    
    return frame

def infer_one_frame(image, model, yolo_model, facial_tracker):
    global eye_closed_start_time, yawn_frame_count, flicker_intensity, flicker_direction, show_warning, frame_count

    eyes_status = ''
    yawn_status = ''
    action = ''

    facial_tracker.process_frame(image)
    if facial_tracker.detected:
        eyes_status = facial_tracker.eyes_status
        yawn_status = facial_tracker.yawn_status

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_result = yolo_model(rgb_image)

    rgb_image = cv2.resize(rgb_image, (224,224))
    rgb_image = tf.expand_dims(rgb_image, 0)
    y = model.predict(rgb_image)
    result = np.argmax(y, axis=1)

    if result[0] == 0 and yolo_result.xyxy[0].shape[0] > 0:
        action = list(ACTIONS.keys())[result[0]]
    if result[0] == 1 and eyes_status == 'eye closed':
        action = list(ACTIONS.keys())[result[0]]

    # Check for eye closure duration
    if eyes_status == 'eye closed':
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        elif time.time() - eye_closed_start_time > EYE_CLOSED_DURATION_THRESHOLD:
            trigger_warning_signal()
    else:
        eye_closed_start_time = None

    # Check for constant yawning
    if yawn_status == 'yawning':
        yawn_frame_count += 1
        if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
            trigger_warning_signal()
    else:
        yawn_frame_count = 0

    cv2.putText(image, f'Driver eyes: {eyes_status}', (30,40), 0, 1,
                conf.LM_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver mouth: {yawn_status}', (30,80), 0, 1,
                conf.CT_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f'Driver action: {action}', (30,120), 0, 1,
                conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

    # Create the warning effect
    image = create_warning_frame(image, flicker_intensity, show_warning, frame_count)
    
    if show_warning:
        # Update flicker intensity
        flicker_intensity += 0.05 * flicker_direction
        if flicker_intensity >= 0.5 or flicker_intensity <= 0:
            flicker_direction *= -1

    frame_count += 1
    
    return image


def check_key_and_reset_warning():
    global show_warning
    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # If any key is pressed
        show_warning = False  # Reset the warning signal
        alert_sound.stop()  # Stop the alert sound

def infer(args):
    image_path = args.image
    video_path = args.video
    cam_id = args.webcam
    checkpoint = args.checkpoint
    save = args.save

    model = MobileNet()
    model.load_weights(checkpoint)

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo_model.classes = [67]

    facial_tracker = FacialTracker()

    if image_path:
        image = cv2.imread(image_path)
        image = infer_one_frame(image, model, yolo_model, facial_tracker)
        cv2.imwrite('images/test_inferred.jpg', image)
    
    if video_path or cam_id is not None:
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(cam_id)
        
        if cam_id is not None:
            cap.set(3, conf.FRAME_W)
            cap.set(4, conf.FRAME_H)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if save:
            out = cv2.VideoWriter('videos/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                fps, (frame_width,frame_height))
        
        while True:
            success, image = cap.read()
            if not success:
                break

            image = infer_one_frame(image, model, yolo_model, facial_tracker)
            check_key_and_reset_warning()  # Check for key press and reset warning if needed
            if save:
                out.write(image)
            else:
                cv2.imshow('DMS', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()
    

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default=None, help='Image path')
    p.add_argument('--video', type=str, default=None, help='Video path')
    p.add_argument('--webcam', type=int, default=None, help='Cam ID')
    p.add_argument('--checkpoint', type=str, help='Pre-trained model file path')
    p.add_argument('--save', type=bool, default=False, help='Save video or not')
    args = p.parse_args()

    infer(args)
