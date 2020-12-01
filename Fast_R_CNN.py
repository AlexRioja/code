import os,cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm


def load_dataset():
    dataset= pickle.loads(open("dataset/data/data.pickle", "rb").read())
    return dataset 

dataset=load_dataset()


"""EMPIEZA LA PARTY
"""
cv2.setUseOptimized(True);
cv2.setNumThreads(4);#no afecta al calculo de segmentación
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# im = cv2.imread("dataset/images/"+dataset[1]['image'])

# ss.setBaseImage(im)
# ss.switchToSelectiveSearchFast()
# rects = ss.process()
# imOut = im.copy()
# for i, rect in (enumerate(rects)):
#     x, y, w, h = rect
# #     print(x,y,w,h)
# #     imOut = imOut[x:x+w,y:y+h]
#     cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
# # plt.figure()
# plt.imshow(imOut)
# plt.show()

#intersection over union
def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


train_images=[]
train_labels=[]


for data in tqdm(dataset):
    image_name=data['image']
    coord=data['data']

    image = cv2.imread("dataset/images/"+image_name) 

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = image.copy()

    counter = 0
    falsecounter = 0
    flag = 0
    fflag = 0
    bflag = 0
    gtvalues=coord

    for e,result in enumerate(ssresults):
        if e < 2000 and flag == 0:
            for gtval in gtvalues:
                x,y,w,h = result
                iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                if counter < 30:
                    if iou > 0.70:
                        timage = imout[y:y+h,x:x+w]
                        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(1)
                        counter += 1
                else :
                    fflag =1
                if falsecounter <30:
                    if iou < 0.3:
                        timage = imout[y:y+h,x:x+w]
                        resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)
                        falsecounter += 1
                else :
                    bflag = 1
            if fflag == 1 and bflag == 1:
                print("inside")
                flag = 1


x_new = np.array(train_images)
y_new = np.array(train_labels)


print(x_new.shape)

with open("dataset/data/X_new.pickle", "wb") as f:
    f.write(pickle.dumps(x_new))
with open("dataset/data/Y_new.pickle", "wb") as f:
    f.write(pickle.dumps(y_new))



