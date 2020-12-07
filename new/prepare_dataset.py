from pascal_voc_writer import Writer
import codecs
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from cv2 import cv2
import pickle


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
    with open("dataset/data/parsed_data.pickle", "wb") as f:
        f.write(pickle.dumps(parsed_data))
    return parsed_data

def save_images_2(parsed_data, debug=False):
    n=0
    print("Creating dataset please wait...")
    parsed_data= pickle.loads(open("dataset/data/parsed_data.pickle", "rb").read())
    for data in tqdm(parsed_data):
        img_path="dataset/images/"
        writer = Writer(img_path+str(n)+'.jpg', 256, 256)
        for d in data[1]:
            points = d['points']
            width=height=256
            if 'Face' in d['label']:
                
                x1 = round(width*points[0]['x'])
                y1 = round(height*points[0]['y'])
                x2 = round(width*points[1]['x'])
                y2 = round(height*points[1]['y'])
                """
                x1 = points[0]['x']
                y1 = points[0]['y']
                x2 = points[1]['x']
                y2 = points[1]['y']
                """
                # ::addObject(name, xmin, ymin, xmax, ymax)
                writer.addObject('face', x1, y1, x2, y2)
        writer.save('dataset/images/data_VOC_pascal'+str(n)+'.xml')
        cv2.imwrite(img_path+str(n)+ ".jpg", cv2.resize(data[0], (256, 256)))
        n+=1
            

eleccion=int(input("Descargar (1) o Parsear(2):"))
if eleccion==1:
    data=load_dataset(path_2_json_dataset)
    parsed_data=parse_data(data, True)
    save_images_2(parsed_data, True)
else:
    save_images_2("",True)