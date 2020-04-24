import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense , Flatten
import numpy as np


x = pickle.load(open("x.pickle","rb")) #shape of x is (number of images , 80 , 80 ,1 ) 80x80 is height and width and 1 is the channel i.e. grayscale here
x = x / 255.0

y = np.asarray(pickle.load(open("y.pickle","rb")))
print(x.shape)

model = Sequential()
model.add( Conv2D(64 ,input_shape=x.shape[1:], kernel_size= (3,3) , padding='same' , strides = (1,1) , activation='relu'))
model.add(  MaxPooling2D(pool_size=(2,2) , padding='valid' , strides = 1)  )

model.add( Conv2D(64 , kernel_size= (3,3) , padding='same' , strides = (1,1) , activation='relu') )
model.add(  MaxPooling2D(pool_size=(2,2) , padding='valid' , strides = 1)  )

model.add(Flatten())
model.add(Dense(1 , activation='sigmoid'))

model.compile(loss = 'binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])

model.fit(x,y,batch_size=32,validation_split=0.1,epochs = 5)

model.save("cat-v-dog.model")