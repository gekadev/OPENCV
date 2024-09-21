import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import os
from datetime import datetime
import logging
from yoloClasses import classNames
from customModelClasses import classNames_
from helper import *
from PIL import Image, ImageDraw, ImageFont
from bidi.algorithm import get_display
import arabic_reshaper
from datetime import datetime
logging.getLogger('ultralytics').setLevel(logging.WARNING)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


########## pathes #########
os.chdir('D:/openCV/vision_ahmed_ibrahim/projects/carPlateRecignition-yolo8')
model = YOLO('D:/openCV/vision_ahmed_ibrahim/projects/smokkeryolov8/yolo-weights/yolov8n.pt')
carPlateModel = YOLO('../carPlateRecignition-yolo8/yolo-weights/charachtersisideplate.pt')
logo = cv2.imread("D:/openCV/vision_ahmed_ibrahim/projects/smokkeryolov8/author/update.png", cv2.IMREAD_UNCHANGED)

########paddleOcr###############
from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='ar')
#############################################

video_path = "../carPlateRecignition-yolo8/videos/video2.mp4"
cap = cv2.VideoCapture(video_path)
wCam, hCam = 1020, 500
cap.set(3, wCam)
cap.set(4, hCam)
track_history = defaultdict(lambda: [])
target_classes = ["car","truck","bus","motorbike"]
target_classes_ = ['car plate']
target_class_ids = [classNames.index(cls_name) for cls_name in target_classes]
target_class_ids_ = [classNames_.index(cls_name_) for cls_name_ in target_classes_]
detectedPlate = set()
fontpath = "arial.ttf"  
font = ImageFont.truetype(fontpath, 32)
font_color = (0, 0, 255)
area = [(11, 264), (12, 364), (1015, 364), (1012, 268)]
#########################

def resumVideo():
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        
#trackers
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('CarPlateRecognition')
cv2.setMouseCallback('CarPlateRecognition', RGB)  

while True:
    ret, frame = cap.read()
    resumVideo()

    if not ret or frame is None:
        print("Failed to capture image")
        break

    frame = cv2.resize(frame, (wCam, hCam))
    results = model.track(source=frame, persist=True, tracker="botsort.yaml")
    carPlateResults = carPlateModel.predict(frame,stream=True)
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
        for box, class_id, track_id in zip(boxes, class_ids, track_ids or []):
               # currentClassName= target_classes[class_id]
                if class_id in target_class_ids :
                    #print(classNames[class_id])
                    cx, cy, w_, h_ = box
                    x1, y1, x2, y2 = getObjectCoordinates(cx, cy, w_, h_)
                    cx=int(cx)
                    cy=int(cy)
                        #start predict car plate
                    for r in carPlateResults :
                        boxs =r.boxes
                        for box in boxs :
                            
                            x1_plate,y1_plate,x2_plate,y2_plate = box.xyxy[0]
                            x1_plate,y1_plate,x2_plate,y2_plate = int(x1_plate),int(y1_plate),int(x2_plate),int(y2_plate)
                            w,h = x2_plate-x1_plate,y2_plate-y1_plate
                            cx_plate,cy_plate = x1_plate+w//2,y1_plate+h//2 
                            conf = math.ceil((box.conf[0]*100))/100
                            cls = int(box.cls[0])
                            currentClass=classNames_[cls]
                            if cls in target_class_ids_:
                                # print(currentClass)
                                if is_full_object_in_area(area, x1_plate, y1_plate, x2_plate, y2_plate):
                                   cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (0, 255, 0), 2)
                                  # cv2.circle(frame, (cx_plate,cy_plate), 5, (255, 0, 255), cv2.FILLED)
                                   #text extraction part 
                                   plate = frame[y1_plate-10:y2_plate, x1_plate:x2_plate]
                                   plate_gray=cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
                                   plate_buler = cv2.GaussianBlur(plate_gray,(3,3),0)
                                   plate_bitral = cv2.bilateralFilter(plate_buler, 10, 20, 20)
                                   plate_thresh = cv2.adaptiveThreshold(plate_bitral,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,11)
                                   result = ocr.ocr(plate_gray, cls=True)
                                   cv2.imshow('plate1',plate)
                                   cv2.imshow('plate2',plate_buler)
                                   cv2.imshow('plate3',plate_bitral)
                                   concatenated_text=''
                                   for line in result:
                                     for word_info in line:
                                        bbox, (text, confidence) = word_info
                                        print(text)
                                        '''
                                        concatenated_text += text +" "                                        
                                        if concatenated_text not in detectedPlate:
                                               detectedPlate.add(concatenated_text)
                                        '''                                                                              
                                        if text not in detectedPlate:
                                            detectedPlate.add(text)   
                                                                          
                                concatenated_plates = ' '.join(list(detectedPlate))
                                concatenated_plates = concatenated_plates[-2:]
                                # Draw arabic letter  on the frame
                                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(img_pil)
                                reshaped_text = arabic_reshaper.reshape(concatenated_plates)
                                bidi_text = get_display(reshaped_text)
                                draw.text((20, 50), bidi_text, font=font, fill=font_color)
                                frame = np.array(img_pil)
                                
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                
                
                #cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
               # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #start detect plate and chracters
                
   # print(detectedPlate)        
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)    
    cv2.imshow('CarPlateRecognition', frame)      

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


