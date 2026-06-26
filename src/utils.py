
import cv2
import numpy as np

def get_transformed_points(boxes, perspective_transform):
    
    bottom_points = []
    for box in boxes:
        pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+box[3])]]] , dtype="float32")
        #pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+(box[3]*0.5))]]] , dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, perspective_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
        bottom_points.append(pnt)
        
    return bottom_points

def cal_dis(p1, p2, width_distance, height_distance):
    
    h = abs(p2[1]-p1[1])
    w = abs(p2[0]-p1[0])
    
    dis_w = float((w/width_distance)*180)
    dis_h = float((h/height_distance)*180)
    
    return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))

def get_distances(boxes1, bottom_points, width_distance, height_distance):
    
    distance_mat = []
    bxs = []
    
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                dist = cal_dis(bottom_points[i], bottom_points[j], width_distance, height_distance)
                if dist <= 150:
                    closeness = 0
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                elif dist > 150 and dist <=180:
                    closeness = 1
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])       
                else:
                    closeness = 2
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                
    return distance_mat, bxs
 
def get_scale(W, H):
    
    dis_w = 400
    dis_h = 600
    
    return float(dis_w/W),float(dis_h/H)
    
def get_count(dist_mat):

    r = []
    g = []
    y = []
    
    for i in range(len(dist_mat)):

        if dist_mat[i][2] == 0:
            if (dist_mat[i][0] not in r) and (dist_mat[i][0] not in g) and (dist_mat[i][0] not in y):
                r.append(dist_mat[i][0])
            if (dist_mat[i][1] not in r) and (dist_mat[i][1] not in g) and (dist_mat[i][1] not in y):
                r.append(dist_mat[i][1])
                
    for i in range(len(dist_mat)):

        if dist_mat[i][2] == 1:
            if (dist_mat[i][0] not in r) and (dist_mat[i][0] not in g) and (dist_mat[i][0] not in y):
                y.append(dist_mat[i][0])
            if (dist_mat[i][1] not in r) and (dist_mat[i][1] not in g) and (dist_mat[i][1] not in y):
                y.append(dist_mat[i][1])
        
    for i in range(len(dist_mat)):
    
        if dist_mat[i][2] == 2:
            if (dist_mat[i][0] not in r) and (dist_mat[i][0] not in g) and (dist_mat[i][0] not in y):
                g.append(dist_mat[i][0])
            if (dist_mat[i][1] not in r) and (dist_mat[i][1] not in g) and (dist_mat[i][1] not in y):
                g.append(dist_mat[i][1])
   
    return (len(r),len(y),len(g))
