from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,GlobalAveragePooling2D,Dropout
import keras as keras
from keras.applications.vgg16 import VGG16
from keras.utils import plot_model

imgsize = 128
base_model = VGG16(include_top=False,
                  input_shape = (imgsize,imgsize,3),
                  weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False
    
for layer in base_model.layers:
    print(layer,layer.trainable)

model = Sequential()
model.add(base_model)
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units = 128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 120, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.models import model_from_json
nn_json = model.to_json()
with open('model_vgg16.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights_vgg16.hdf5"
model.save_weights(weights_file,overwrite=True)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
