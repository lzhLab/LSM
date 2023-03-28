#deal 3D segmention datasets
import os
import os.path as osp
from os.path import join
import random
from tqdm import tqdm
import torch
import cv2
import numpy as np
import math
from  math import pi,e


def crop_mask(image,center, rediuns):
    roi = cv2.circle(np.zeros_like(image), center, rediuns, 1, cv2.FILLED)
    roi = image * roi
    return roi

def waijie_juxing1(img):    
    conts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(conts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    
    box = np.int0(box)
    #print('waijie point:', box)
    x1 = min(box[0][0], box[3][0])
    y1 = min(box[0][1], box[1][1])
    x2 = max(box[2][0], box[3][0])
    y2 = max(box[0][1], box[3][1])
    return x1,y1,x2,y2


def norm_sampling1(search_space, image):
    flag = 0
    while flag <1:
        search_x_left, search_y_left, search_x_right, search_y_right = search_space
        
        new_bbox_x_center = random.randint(search_x_left, search_x_right)
        new_bbox_y_center = random.randint(search_y_left, search_y_right)
        if image[new_bbox_y_center][new_bbox_x_center] != 0:
            flag +=1
    return [new_bbox_x_center, new_bbox_y_center]


def mask_iou(paste_mask, mask):
    return (paste_mask*mask).sum()

def gauss(dilate_num,x):
    var = (math.floor(dilate_num/2)+1)**2/(math.log(2,e))
    f = math.exp(-x**2/var)
    return f

def get_slices(slicenames, big_zhixin_list):
    allres = []
    
    for m in range(len(big_zhixin_list)):
        big_zhixin_list[m].append([])
    for fi in slicenames:
        mask = cv2.imread(fi, 0)
        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, 4)
        for j in range(len(contours)):
            singer_contours = contours[j]
            if len(singer_contours.reshape(-1, 2).tolist()) > 2:
                bounding_boxes = cv2.boundingRect(singer_contours)
                x1 = bounding_boxes[0]
                y1 = bounding_boxes[1]
                x2 = x1+bounding_boxes[2]
                y2 = y1+bounding_boxes[3]
                center_x = x1+math.ceil(bounding_boxes[2]/2)
                center_y = y1+math.ceil(bounding_boxes[3]/2)
                tmp = 10000
                flag = 0
                #判断当前病变质心距离哪个大质心更近，就归为哪一个区域
                for m in range(len(big_zhixin_list)):
                    x_big = big_zhixin_list[m][0][0]
                    y_big = big_zhixin_list[m][0][1]
                    juli = math.hypot(x_big - center_x, y_big - center_y)
                    if juli < tmp:
                        tmp = juli
                        flag = m
                big_zhixin_list[flag][2].append(fi)
    return big_zhixin_list

                
def count_points(slicenames, enlarge, save_base_dir, save_mask_dir):

    iou_thresh=0
    for i in range(len(slicenames)):
        #i = 3
        #print(slicenames[i])
        big_img = np.zeros((512, 512))
        for fi in slicenames[i]:
            mask = cv2.imread(fi, 0)
            big_img = big_img + mask
        big_img[big_img > 1] = 1
        big_img = np.array(big_img).astype(np.uint8)
        contours, hier = cv2.findContours(big_img * 255, cv2.RETR_CCOMP, 4)
        #每个silcenames 做并集，存储这一小段连续序列的质心
        big_zhixin_list = []
        for j in range(len(contours)):
            singer_contours = contours[j]
            if len(singer_contours.reshape(-1, 2).tolist()) > 2:
                #point = np.array(singer_contours).reshape(-1, 2).mean(0).astype(np.int).tolist()
                
                bounding_boxes = cv2.boundingRect(singer_contours)
                x1 = bounding_boxes[0]
                y1 = bounding_boxes[1]
                x2 = x1+bounding_boxes[2]
                y2 = y1+bounding_boxes[3]
                center_x = x1+math.ceil(bounding_boxes[2]/2)
                center_y = y1+math.ceil(bounding_boxes[3]/2)
                radius = math.ceil(max(bounding_boxes[2], bounding_boxes[3])/2)+1
                big_zhixin_list.append([(center_x, center_y),radius])
                #print(center_x, center_y, radius)
        
        #print(big_zhixin_list)
        big_zhixin_list_slices = get_slices(slicenames[i], big_zhixin_list)
        print(big_zhixin_list_slices)
        #break
        
        new_big_zhixin_list = big_zhixin_list.copy()
        
        #判断每个层病灶外扩后是否侵占其他病灶
        new_centers = []
        for center_radius in big_zhixin_list_slices:
            #print(center_radius)
            center_x = center_radius[0][0]
            center_y = center_radius[0][1]
            radius = center_radius[1]
            zhixin_slices = center_radius[2]
            iscrop = True
            tmp_liver = np.zeros((512,512))
            
            for fi in slicenames[i]:
                mask = cv2.imread(fi, 0)/255
                liver = cv2.imread(fi.replace('trainmask', 'train'), 0)
                cropliver = crop_mask(liver, (center_x, center_y), radius)
                cropmask = crop_mask(mask, (center_x, center_y), radius+1)
                
                kernel = np.ones((3,3), np.uint8)
                mask_ori_dilate1 = cv2.dilate(cropmask, kernel, 1)
                mask_ori_dilate2 = cv2.dilate(mask_ori_dilate1, kernel, 1)
                mask_ori_dilate3 = cv2.dilate(mask_ori_dilate2, kernel, 1)
                mask_ori_dilate4 = cv2.dilate(mask_ori_dilate3, kernel, 1)
                mask_ori_dilate5 = cv2.dilate(mask_ori_dilate4, kernel, 1)
                mask_ori_dilate6 = cv2.dilate(mask_ori_dilate5, kernel, 1)
                mask_ori_dilate7 = cv2.dilate(mask_ori_dilate6, kernel, 1)
                if (mask_ori_dilate7*mask).sum()==cropmask.sum() and cropmask.sum()<1024:
                    tmp_liver = tmp_liver + liver
                else:
                    iscrop = False
            print(iscrop)
            if iscrop:
                
                #在开始的第一张上选择粘贴位置
                center_search_space = waijie_juxing1(cv2.imread(zhixin_slices[0].replace('trainmask', 'train'),0))#debug，如果连续病灶不从第一张开始
                #print(center_search_space)
                x1 = center_search_space[0]
                y1 = center_search_space[1]
                x2 = center_search_space[2]
                y2 = center_search_space[3]
                if y2-y1>=radius and x2-x1>=radius:
                    flag = 0
                    success_num = 0
                    paste_number=1
                    success_slices = 0#
                    while success_num < paste_number:
                        liver = cv2.imread(slicenames[i][0].replace('trainmask', 'train'), 0)
                        #判断每层粘贴位置(以第一层粘贴位置为准)是否合适,
                        new_bbox_center = norm_sampling1(center_search_space, liver)   # 随机生成点坐�?    
                        flag+=1
                        #print(flag,new_bbox_center)
                        if flag>10:
                            break
                        for fi in slicenames[i]:
                            #print(fi)
                            liver = cv2.imread(fi.replace('trainmask', 'train'), 0)
                            livermask = np.where(liver>0, 1, 0).astype(np.uint8)
                            
                            mask = cv2.imread(fi, 0)/255
                            paste_mask = crop_mask(livermask, (new_bbox_center[0], new_bbox_center[1]), round(enlarge+radius)) 
                          
                            if (paste_mask*mask).sum() != 0:
                                continue
                            if paste_mask.sum()< pi*(radius+enlarge)**2/2:
                                continue
                            if new_bbox_center[0] + radius > center_search_space[2] :
                                continue
                            if new_bbox_center[1] + radius >center_search_space[3]:
                                continue  
                            #print('iou:', new_big_zhixin_list)
                            ious = [mask_iou(crop_mask(livermask, center, round(enlarge+redius1)), paste_mask) for center, redius1, _ in new_big_zhixin_list]
                            #print(ious)
                            if max(ious) <= iou_thresh:
                                success_num += 1
                                success_slices+=1
                            else:
                                continue
                    print('new_bbox_center', new_bbox_center)
                    if success_slices == len(slicenames[i]):
                        new_centers.append([new_bbox_center[0],new_bbox_center[1],center_x,center_y, radius])
                        new_big_zhixin_list.append([(new_bbox_center[0], new_bbox_center[1]), enlarge+radius, []])
            
        print(new_centers)
        if len(new_centers) !=0:
            tmp_res = set()
            for fi in slicenames[i]:
                image = cv2.imread(fi.replace('trainmask', 'train'), 0)
                liver_mask = np.where(image>0, 1,0)
                mask = cv2.imread(fi, 0)
                image1 = image.copy()
                mask1 = mask.copy()

                for k, center  in enumerate(new_centers):
                    w_c = center[0] - center[2]
                    h_c = center[1] - center[3]
                    radius = center[4]
                    center_x = center[2]
                    center_y = center[3]
                    crop_ori = crop_mask(image, (center_x, center_y), radius)
                    mask_ori = crop_mask(mask, (center_x, center_y), radius)
                    kernel = np.ones((3,3), np.uint8)

                    mask_ori_dilate1 = cv2.dilate(mask_ori, kernel, 1)
                    mask_ori_dilate2 = cv2.dilate(mask_ori_dilate1, kernel, 1)
                    mask_ori_dilate3 = cv2.dilate(mask_ori_dilate2, kernel, 1)
                    mask_ori_dilate4 = cv2.dilate(mask_ori_dilate3, kernel, 1)
                    mask_ori_dilate5 = cv2.dilate(mask_ori_dilate4, kernel, 1)
                    mask_ori_dilate6 = cv2.dilate(mask_ori_dilate5, kernel, 1)
                    mask_ori_dilate7 = cv2.dilate(mask_ori_dilate6, kernel, 1)
                    
                    mask_ori_dilate7 = mask_ori_dilate7*liver_mask
                    for i in range(center_x-radius-enlarge, center_x+radius+enlarge):
                        for j in range( center_y-radius-enlarge,  center_y+radius+enlarge):
                            if mask_ori_dilate7[j][i]!=0:
                                ni = j+h_c 
                                nj = i+w_c
                                newpoint = (nj,ni)
                                if image[newpoint[1]][newpoint[0]]!=0:
                                    if mask_ori[j][i] ==0:
                                        dist = 0
                                    
                                        if mask_ori_dilate1[j][i]!=0:
                                            dist = 1
                                        elif mask_ori_dilate2[j][i]!=0:
                                            dist = 2
                                        elif mask_ori_dilate3[j][i]!=0:
                                            dist = 3
                                        elif mask_ori_dilate4[j][i]!=0:
                                            dist = 4
                                        elif mask_ori_dilate5[j][i]!=0:
                                            dist = 5
                                        elif mask_ori_dilate6[j][i]!=0:
                                            dist = 6
                                        elif mask_ori_dilate7[j][i]!=0:
                                            dist = 7
                                    
                                        dis = gauss(enlarge, dist)
                                        image1[newpoint[1]][newpoint[0]] =  round(dis*image[j][i]+(1-dis)*image[newpoint[1]][newpoint[0]])
                                        
                                        #reverse gauss
                                        #image1[newpoint[1]][newpoint[0]] = round((1-dis)*image[j][i]+dis*image[newpoint[1]][newpoint[0]])
                                    else:
                                        image1[newpoint[1]][newpoint[0]] = image[j][i]
                                else:
                                    if mask_ori[j][i]!=0:
                                        image1[newpoint[1]][newpoint[0]] = image[j][i]
                    
                    for i in range(center_x-radius, center_x+radius):
                        for j in range(center_y-radius, center_y+radius):
                            if mask_ori[j][i]!=0:
                                newpoint =[ i+w_c, j+h_c ]
                                mask1[newpoint[1]][newpoint[0]] = 255
                   
                savename = save_base_dir+'/'+fi.split('/')[-2]
                if not osp.exists(savename):
                    os.makedirs(savename)
                cv2.imwrite(osp.join(savename, fi.split('/')[-1]), image1)

                savemaskname =  save_mask_dir+'/'+fi.split('/')[-2]
                if not osp.exists(savemaskname):
                    os.makedirs(savemaskname)
                cv2.imwrite(osp.join(savemaskname, fi.split('/')[-1]), mask1)
        else:
            for fi in slicenames[i]:
                image = cv2.imread(fi.replace('trainmask', 'train'), 0)
                mask = cv2.imread(fi, 0)
                savename = save_base_dir+'/'+fi.split('/')[-2]
                if not osp.exists(savename):
                    os.makedirs(savename)
                cv2.imwrite(osp.join(savename, fi.split('/')[-1]), image)

                savemaskname =  save_mask_dir+'/'+fi.split('/')[-2]
                if not osp.exists(savemaskname):
                    os.makedirs(savemaskname)
                cv2.imwrite(osp.join(savemaskname, fi.split('/')[-1]), mask)

num = 0
enlarge = 7
base = './Ply_Gaussian_{}_3D'.format(enlarge)
save_base_dir = join(base, 'train')
save_mask_dir = join(base, 'trainmask')
if not osp.exists(base):
    os.mkdir(base)
    os.mkdir(save_base_dir)
    os.mkdir(save_mask_dir)

paths = ['./data/train']
for path in paths:
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            has_saved = osp.join(save_base_dir, dir)
            if not osp.exists(has_saved):
                dirpath = osp.join(root, dir)
                trainfiles = []
                maskfiles = []
                for fil in os.listdir(dirpath):
                    if fil.endswith('png'):
                        trainfiles.append(osp.join(dirpath, fil))
                        if osp.exists(osp.join(dirpath, fil).replace('train', 'trainmask')):
                            maskfiles.append(osp.join(dirpath, fil).replace('train', 'trainmask'))
                sort_trainfiles = sorted(trainfiles, key=lambda x: int(x.split('.')[-2]))
                liver = []
                mask = []
                for fil in sort_trainfiles:
                    liver.append(cv2.imread(fil, 0))
                    if osp.exists(fil.replace('train', 'trainmask')):
                        mask.append(cv2.imread(fil.replace('train', 'trainmask'), 0))
                    else:
                        mask.append(np.zeros((512,512)))
                
                
                sort_maskfiles = sorted(maskfiles, key=lambda x: int(x.split('.')[-2]))
                slicenames = []
                for i in range(len(sort_maskfiles)):
                    if not slicenames:
                        slicenames.append([sort_maskfiles[i]])
                    elif int(sort_maskfiles[i-1].split('.')[-2])+1 == int(sort_maskfiles[i].split('.')[-2]):
                        slicenames[-1].append(sort_maskfiles[i])
                    else:
                        slicenames.append([sort_maskfiles[i]])

                count_points(slicenames, enlarge, save_base_dir, save_mask_dir)




