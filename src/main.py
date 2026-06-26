#!/usr/bin/env python

import cv2
import numpy as np
import time
import argparse


import utils, plot

confid = 0.5
thresh = 0.5
ms_points = []

def get_mouse_points(event, x, y, flags, param):

    global ms_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ms_points) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(ms_points) >= 1 and len(ms_points) <= 3:
            cv2.line(image, (x, y), (ms_points[len(ms_points)-1][0], ms_points[len(ms_points)-1][1]), (70, 70, 70), 2)
            if len(ms_points) == 3:
                cv2.line(image, (x, y), (ms_points[0][0], ms_points[0][1]), (70, 70, 70), 2)
        
        ms_points.append((x, y))
      
        


def calculate_cnn_social_distancing(video_path, net, output_dir, output_vid, ln1):
    
    count = 0
    vs = cv2.VideoCapture(video_path)    


    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
   
    scale_w, scale_h = utils.get_scale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    bird_video = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
        
    points = []
    global image
    
    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            break
            
        (H, W) = frame.shape[:2]
        
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(ms_points) == 8:
                    cv2.destroyWindow("image")
                    break
               
            points = ms_points      
                 
       
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        perspective_transform = cv2.getPerspectiveTransform(src, dst)

        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, perspective_transform)[0]
        
        width_distance = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        height_distance = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
    ####################################################################################
    
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
       
        pedestrians_pts = utils.get_transformed_points(boxes1, perspective_transform)
        
       
        dist_mat, boxes_mat = utils.get_distances(boxes1, pedestrians_pts, width_distance, height_distance)
        risk_count = utils.get_count(dist_mat)
    
        frame1 = np.copy(frame)
        
        birdeyeview_image = plot.bird_eye_view(frame, dist_mat, pedestrians_pts, scale_w, scale_h, risk_count)
        img = plot.social_distancing_view(frame1, boxes_mat, boxes1, risk_count)
        
        if count != 0:
            output_video.write(img)
            bird_video.write(birdeyeview_image)
    
            cv2.imshow('Bird Eye View', birdeyeview_image)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, birdeyeview_image)
    
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example.mp4' ,
                    help='Path for input video')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    
    parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default='./output_vid/' ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='../models/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
        
    output_dir = values.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'
    
    output_vid = values.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'


    
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"
    
    
    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i - 1] for i in net_yl.getUnconnectedOutLayers()]


    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)
    
    calculate_cnn_social_distancing(values.video_path, net_yl, output_dir, output_vid, ln1)



