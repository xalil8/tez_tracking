
#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from trackers.multi_tracker_zoo import create_tracker

start_time = time()

_create_default_https_context = _create_unverified_context


source_video_path = "demo_videos/berk_demo.mp4"
video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"


model = torch.hub.load('ultralytics/yolov5', "yolov5m",force_reload=False)
model.to(torch.device("mps"))

model.classes = [2,5,7]
names = model.names #to get class names 
#model.classes = [None]
vide_write = True

#to use with default yolo model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_cap=cv2.VideoCapture(source_video_path)

fps = video_cap.get(cv2.CAP_PROP_FPS)
width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

desired_fps = 35

if vide_write:
    result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))

########################POLYGON############################

polygon_points= np.array(  [[1054, 14], [1134, 12], [1872, 773], [1428, 769]])
passing_dict= {}
object_counter = 0

########################TRACKER###########################
tracker_list = []
tracker = create_tracker('bytetrack', "trackers/bytetrack/configs/bytetrack.yaml", "weights/osnet_x0_25_msmt17.pt", torch.device("mps"),False)
tracker_list.append(tracker, )
object_counter = 0

count=0
while video_cap.isOpened():

    ret,frame=video_cap.read()

    if not ret:
        break
    #ADJUST FPS
    count +=1
    if count % 1 != 0:
        continue

    results = model(frame)
    det = results.xyxy[0]
    cv2.putText(frame, f"{object_counter} CAR IN SELECTED LINE ", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.polylines(frame, np.int32([polygon_points]), True, (255,0,0),3)

    if det is not None and len(det): #work if there is detections
        outputs = tracker_list[0].update(det.cpu(), frame)
        
        for j, (output) in enumerate(outputs):
            bbox = output[0:4]
            id = output[4]
            cls = output[5]
            conf = output[6]
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2, c, id= int(x1),int(y1),int(x2), int(y2), int(cls), int(id)
            ###############################ALGORITHM WORK HERE###################################
            center_x, center_y= int((x1+x2)/2), int((y1+y2)/2)
            area_check_1 = cv2.pointPolygonTest(np.int32([polygon_points]),((center_x,center_y)), False)


            if area_check_1 == 1:
                in_color = (50,50,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)

            else:
                in_color = (50,255,50)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)


            if id not in passing_dict:
                passing_dict[id] = "new"


            if area_check_1 == 1:
                if passing_dict[id] == "new":
                    passing_dict.update({id:"in"})
                    object_counter += 1
                    
            else:
                if passing_dict[id] == "in":
                    passing_dict.update({id:"out"})
                    object_counter -= 1


    cv2.imshow("ROI",frame)
    if vide_write:
        print(f"frame {count} writing")
        result.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break


video_cap.release()
if vide_write:
    result.release()
cv2.destroyAllWindows()
print("process done")
print("Execution time:", time() - start_time, "seconds")
