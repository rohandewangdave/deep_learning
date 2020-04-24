import cv2
import tensorflow as tf

categories = ["Dog","Cat"]
def prepare(filepath):
    img_size = 80
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)
model = tf.keras.models.load_model('cat-v-dog.model')
predicition = model.predict([prepare("dog.jfif")])
print(categories[int(predicition[0][0])])
