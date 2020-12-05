import numpy as np
import pandas as pd 
import json
import codecs
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import os
import pickle
from cv2 import cv2

path_2_json_dataset="dataset/face_detection.json"

def load_dataset(dataset_path, debug=False):
    """Loads the dataset in json format to a list and then returns it

    Args:
        dataset_path (str): Path to the .json dataset file
        debug (bool, optional): Set True to see debug info like image count or sample data. Defaults to False.

    Returns:
        list: list containing on each position, the information of an image
    """
    data=[]
    with codecs.open(path_2_json_dataset, 'rU', 'utf-8') as json_file:
        for line in json_file:
            data.append(json.loads(line))
    if debug:
        print("Dataset containing: "+str(len(data))+" images!")
        print(data[0])
    return data


def parse_data(data, debug=False):
    """Parse data from the return of load_dataset

    Args:
        data (list): list containing on each position, the information of an image
        debug (bool, optional): Set True to see debug info like image count or sample data. Defaults to False.

    Returns:
        list: Containing the information of the image and the data related to it.
    """
    parsed_data=[]
    print("\nPlease wait while the images are being downloaded...")
    for row in tqdm(data):
        image_response = requests.get(row['content'])
        img = np.asarray(Image.open(BytesIO(image_response.content)))
        parsed_data.append([img, row["annotation"]])
    print("All images downloaded now!\n")
    if debug:
        print("Parsed dataset containing: "+str(len(parsed_data))+" images!")
        print(parsed_data[0])
    return parsed_data


def save_images(parsed_data, debug=False):
    try:
        os.mkdir("dataset/images/")
    except OSError as error:
        print(error)
        pass
    n=0
    data_correlation=[]
    print("Creating dataset please wait...")
    for data in tqdm(parsed_data):
        img_path="dataset/images/"
        
        for d in data[1]:
            height = 300
            width = 300
            points = d['points']
            t=0
            if 'Face' in d['label']:
                x1 = round(width*points[0]['x'])
                y1 = round(height*points[0]['y'])
                x2 = round(width*points[1]['x'])
                y2 = round(height*points[1]['y'])
                aux={"x1":x1,"y1":y1,"x2":x2,"y2":y2}
                data_correlation.append({"image":str(n)+'_'+str(t)+ ".jpg", "data":aux})
                cv2.imwrite(img_path+str(n)+'_'+str(t)+ ".jpg", cv2.resize(data[0], (300, 300)))
                t+=1
        n+=1
    with open("dataset/data/data.pickle", "wb") as f:
        f.write(pickle.dumps(data_correlation))
    if debug:
        print("Example of information associated to an image: ")
        print(data_correlation[0])
    

def save_images_2(parsed_data, debug=False):
    try:
        os.mkdir("dataset/images/")
    except OSError as error:
        print(error)
        pass
    n=0
    data_correlation=[]
    print("Creating dataset please wait...")
    for data in tqdm(parsed_data):
        img_path="dataset/images/"
        aux=[]
        for d in data[1]:
            height = 256
            width = 256
            points = d['points']
            
            if 'Face' in d['label']:
                x1 = round(width*points[0]['x'])
                y1 = round(height*points[0]['y'])
                x2 = round(width*points[1]['x'])
                y2 = round(height*points[1]['y'])
                aux2={"x1":x1,"y1":y1,"x2":x2,"y2":y2}
                aux.append(aux2)
        data_correlation.append({"image":str(n)+ ".jpg", "data":aux})
        cv2.imwrite(img_path+str(n)+ ".jpg", cv2.resize(data[0], (300, 300)))
        n+=1
    with open("dataset/data/data.pickle", "wb") as f:
        f.write(pickle.dumps(data_correlation))
    if debug:
        print("Example of information associated to an image: ")
        print(data_correlation[0])
    return data_correlation


def write_line(img, data):
    A=4
    B=5
    id=1 #solo tenemos una clase (cara)
    bboxes=[]
    for d in data:
        x1=d['x1']
        x2=d['x2']
        y1=d['y1']
        y2=d['y2']
        bboxes.append(d.items())
        

def create_RecordIO_format():
    img_path="dataset/images/"
    data_correlation= pickle.loads(open("dataset/data/data.pickle", "rb").read())
    print("Formatting to RecordIO...")
    with open('train.lst', 'w+') as f:  
        for img, data in tqdm(enumerate(data_correlation)):





            f.write(  
                str(i) + '\t' +  
                # idx  
                str(4) + '\t' + str(5) + '\t' +  
                # width of header and width of each object.  
                str(256) + '\t' + str(256) + '\t' +  
                # (width, height)  
                str(1) + '\t' +  
                # class  
                str((i / 10)) + '\t' + str((i / 10)) + '\t' + str(((i + 3) / 10)) + '\t' +str(((i + 3) / 10)) + '\t' +  
                # xmin, ymin, xmax, ymax  
            str(i) + '.jpg\n'
            )


eleccion=input("Descargar (1) o Parsear(2):")
if eleccion==1:
    data=load_dataset(path_2_json_dataset)
    parsed_data=parse_data(data, True)
    data_correlation=save_images_2(parsed_data, True)
else:
    create_RecordIO_format()

print("\n\nDataset is ready!!!")
