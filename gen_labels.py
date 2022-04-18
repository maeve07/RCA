import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists
from utils import imutils

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

# prepare data
data_path = 'VOC2012/'
train_lst_path = data_path + 'ImageSets/Segmentation/train_cls.txt'
im_path = data_path + 'JPEGImages/'
sal_path = data_path + 'saliency_map'
att_path = 'orig/'
save_path ='data/proxy_label/' 

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

def _crf_with_alpha(image, cam_dict, alpha, t=10):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = imutils.crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t)
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1] #+1
    return n_crf_al

# generate proxy labels
def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    
    im_name = im_path + name + '.jpg'
    bg_name = sal_path + name + '.png'

    img = cv2.imread(im_name)
    sal = cv2.imread(bg_name, 0)
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    sal = np.array(sal, dtype=np.float32)
    
    conflict = 0.9
    bg_thr = 32
    att_thr = 0.8

    gt[0] = (1 - (sal / 255))
    init_gt = np.zeros((height, width), dtype=float) 
    sal_att = sal.copy()  
    cam_dict = {}
    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        cam_dict[cls] = att
    alpha = 4
    t = 10
    cam_crf = _crf_with_alpha(img, cam_dict, alpha, t)

    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])    
        att = cam_crf.get(cls+1)
        gt[cls+1] = att  #.copy()
        sal_att = np.maximum(sal_att, ((np.array(att)) > att_thr) *255)

    bg = np.array(gt > conflict, dtype=np.uint8)  
    bg = np.sum(bg, axis=0)
    gt = gt.argmax(0).astype(np.uint8) 
    gt[bg > 1] = 255

    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255

    out = gt 
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

for i in range(len(lines)):
   gen_gt(i)

