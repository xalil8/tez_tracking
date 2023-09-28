#TODO make polygons filled with transparant color

import torch
import cv2
import numpy as np
from trackers.multi_tracker_zoo import create_tracker
source_video_path = "kayit2.mp4"
model = torch.hub.load('ultralytics/yolov5', "custom",'tez.pt',
                       force_reload=False,device="mps")
names = model.names #to get class names 
video_cap=cv2.VideoCapture(source_video_path)
fps = video_cap.get(cv2.CAP_PROP_FPS)
width=  int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
########################POLYGON############################
polygon_points= np.array([[283, 393],[484, 267],[551, 375],[305, 524]])
polygon_points2 = np.array([[564, 391], [307, 551],[342, 754],[663, 547]])
in_polyline = np.array([[564, 391],[307, 551]])
out_polyline = np.array([[305, 524],[551, 375]])
passing_dict= {}
outcounter = 0
incounter = 0
########################TRACKER###########################
tracker_list = []
tracker = create_tracker('bytetrack', 
                         "trackers/bytetrack/configs/bytetrack.yaml",
                         "weights/osnet_x0_25_msmt17.pt", 
                         device=torch.device("mps"),half =False)
tracker_list.append(tracker, )
########################MAIN LOOP###########################
count=0
while video_cap.isOpened():
    ret,frame=video_cap.read()
    if not ret:break
    results = model(frame)
    det = results.xyxy[0]
    cv2.putText(frame, f"{incounter} IN ", (30,50),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,229,204), 3)
    cv2.putText(frame, f"{outcounter} OUT ", (30, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,229,204), 3)
    cv2.polylines(frame, np.int32([polygon_points]), True, (255,0,0),3)
    cv2.polylines(frame, np.int32([polygon_points2]), True, (255, 0, 0), 3)
    cv2.polylines(frame, np.int32([in_polyline]), True, (255,155,255),3)
    cv2.polylines(frame, np.int32([out_polyline]), True, (255, 155, 255), 3)
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
            center_x, center_y= int((x1+x2)/2), int(y2-10)
            area_check_1 = cv2.pointPolygonTest(np.int32([polygon_points]),
                                                ((center_x,center_y)), False)
            area_check_2 = cv2.pointPolygonTest(np.int32([polygon_points2]),
                                                 ((center_x, center_y)), False
            if area_check_1 == 1:
                in_color = (50,255,50)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)
            else:
                in_color = (50,255,50)
                cv2.rectangle(frame,(x1,y1),(x2,y2),in_color,2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)
            if area_check_2 == 1:
                in_color = (50,255,50)
                cv2.rectangle(frame, (x1, y1), (x2, y2), in_color, 2)
                cv2.circle(frame, (center_x, center_y), radius=3,
                           color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)
            else:
                in_color = (50,255,50)
                cv2.rectangle(frame, (x1, y1), (x2, y2), in_color, 2)
                cv2.circle(frame, (center_x, center_y), radius=3,
                           color=in_color, thickness=-1)
                cv2.putText(frame, f"{names[int(c)]}{str(id)}", (x1,y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, in_color, 2)
            if id not in passing_dict:
                passing_dict[id] = "new"
            print(passing_dict)
            if area_check_1 == 1:
                if passing_dict[id] == "new":
                    passing_dict.update({id:"inx"})
                    #object_counter += 1
                if passing_dict[id] == "iny":
                    passing_dict.update({id: "inx"})
                    outcounter += 1
            
            if area_check_2 == 1:
                if passing_dict[id] == "new":
                    passing_dict.update({id: "iny"})
                    #object_counter += 1
                if passing_dict[id] == "inx":
                    passing_dict.update({id: "iny"})
                    incounter += 1
    cv2.imshow("ROI",frame)
    if cv2.waitKey(20) == ord('q'):
        break
cv2.destroyAllWindows()