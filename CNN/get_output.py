from keras.models import load_model
import numpy
import cv2

#C:\Users\user\Downloads\pred\pred0.jpg
target_dict = {0:"normal",1:"Tumer"}

file_path = input("enter a file path : ") #enter a brain photo :)
model  = load_model("CNN\model.keras")

image = cv2.imread(file_path)
image = cv2.resize(image,(224,224))
test_img=image.reshape(1,224,224,3)
image = numpy.array(test_img)
new_image = image/255

predict = model.predict(new_image)

result = numpy.argmax(predict,axis=1)[0]
acc=numpy.max(predict,axis=1)[0]

lable = target_dict[result]

print(lable)
print(f"accurncy:{acc}")
