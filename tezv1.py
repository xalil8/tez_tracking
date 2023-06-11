
#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
from ssl import _create_unverified_context
from time import time
from trackers.multi_tracker_zoo import create_tracker

start_time = time()

_create_default_https_context = _create_unverified_context


source_video_path = "tez.mp4"
video_saving_path = source_video_path[:len(source_video_path)-4:]+"_output.mp4"


#model = torch.hub.load('ultralytics/yolov5', "yolov5m",force_reload=False)
model = torch.hub.load('ultralytics/yolov5', "custom",'tez.pt',force_reload=False,device="mps")

model.to(torch.device("mps"))
names = model.names #to get class names 
model.classes = [0]

#to use with default yolo model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

video_cap=cv2.VideoCapture(source_video_path)

fps = video_cap.get(cv2.CAP_PROP_FPS)
width, height = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

desired_fps = 35
result = cv2.VideoWriter(video_saving_path, cv2.VideoWriter_fourcc(*'mp4v') ,fps, (width,height))

########################POLYGON############################
polygon_points= np.array( [[804, 9], [27, 780], [456, 805], [879, 13]])

polygon_points_2= np.array( [[804, 9], [27, 780], [456, 805], [879, 13]])
passing_dict= {}
object_counter = 0

########################TRACKER###########################
tracker_list = []
tracker = create_tracker('bytetrack', "trackers/bytetrack/configs/bytetrack.yaml", "weights/osnet_x0_25_msmt17.pt", torch.device("mps"),False)
tracker_list.append(tracker, )

count=0
while video_cap.isOpened():
    ret,frame=video_cap.read()

    if not ret:
        break
    #ADJUST FPS
    count +=1
    if count % 1 != 0:
        continue

    cv2.putText(frame, f"{object_counter} CAR IN THIRD LINE ", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    #cv2.polylines(frame, np.int32([polygon_points]), True, (255,0,0),3)

    results = model(frame)
    det = results.xyxy[0]

    if det is not None and len(det): #work if there is detections
        outputs = tracker_list[0].update(det.cpu(), frame)
                
        for j, (output) in enumerate(outputs):
            #print(output)
            bbox = output[0:4]
            id = output[4]
            cls = output[5]
            conf = output[6]
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2, c, id= int(x1),int(y1),int(x2), int(y2), int(cls), int(id)
            ###############################ALGORITHM WORK HERE###################################
            center_x, center_y= int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)
            #cv2.rectangle(frame,(x1,y1),(x2,y2),(20,0,255),2)
            cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,255,20), 2)
            cv2.putText(frame, f"conf %{str(round(conf*100,0))}", (x1,y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,255,20), 2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(20,0,255),2)







    cv2.putText(frame, "By COPY PASTE AI ", (width-250,height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
    cv2.imshow("ROI",frame)

    #print(f"frame {count} writing")
    result.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break


video_cap.release()
result.release()
cv2.destroyAllWindows()
print("process done")
print("Execution time:", time() - start_time, "seconds")
