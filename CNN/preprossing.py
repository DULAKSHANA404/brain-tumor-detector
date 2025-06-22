import cv2
import os
import numpy as np
from keras.utils import to_categorical
print("start")

file_path =r"C:\Users\user\Desktop\brain_tumer\CNN\train"
catagories = os.listdir(file_path)
lable = np.arange(len(catagories))
dict = dict(zip(catagories,lable))  #{'normal': 0, 'tumer': 1}

data = []
target = []

for catagorie in catagories:
    image_paths = os.path.join(file_path,catagorie)    
    path = os.listdir(image_paths)

    for image in path:
        image_path = os.path.join(image_paths,image)

        try :
            image = cv2.imread(image_path)
            image = cv2.resize(image,(224,224))

            data.append(image)
            target.append(dict[catagorie])
            
        except:
            pass
        
data = np.array(data)
target  = np.array(target)
data_new = data/255
target_new = to_categorical(target)
print(target.shape)
print(data_new.shape)

np.save("data.npy",data_new)
np.save("target.npy",target_new)

print("suck*seed")

