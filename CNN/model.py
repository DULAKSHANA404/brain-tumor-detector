from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D,Flatten,Dense,Input,Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np

data_path = r"CNN\data.npy"
target_path =r"CNN\target.npy"

data = np.load(data_path)
target = np.load(target_path)

base_model = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

headmodel = base_model.output
headmodel = AveragePooling2D(pool_size=(4,4))(headmodel)
headmodel = Flatten(name = "flatten")(headmodel)
headmodel = Dense(128,activation="relu")(headmodel)
headmodel = Dropout(0.5)(headmodel)
headmodel = Dense(2,activation="softmax")(headmodel)

model = Model(inputs = base_model.input,outputs = headmodel)

for layer in base_model.layers:
    layer.trainable= False

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
   
train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.2)

model.fit(train_data,train_target,validation_data=(test_data,test_target),epochs=10)

model.save("model.keras")