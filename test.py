import model as net
import pickle
from cv2 import cv2

def load_dataset():
    dataset= pickle.loads(open("dataset/data/data.pickle", "rb").read())
    return dataset 

dataset=load_dataset()
print(dataset[1])

image = cv2.imread("dataset/images/"+dataset[1]['image'], 0)

coord=dataset[1]['data']
print(coord)
cv2.rectangle(image, (coord['x1'], coord['y1']), (coord['x2'], coord['y2']),(255, 0, 0), 2)

cv2.imshow("Test", image)
cv2.waitKey(10000)
#print(image.shape)

# conv = net.Conv3x3(8)
# pool = net.MaxPool2()

# output = conv.forward(image)
# output = pool.forward(output)
# print(output.shape)
# network=net.Net()
# out, l, acc=network.forward(image,2)