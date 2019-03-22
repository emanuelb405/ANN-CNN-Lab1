import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Lambda
import keras as keras
from keras import backend as K

#generate network
#model = Sequential()
#model.add(Conv2D(32,3,strides=3,input_shape=(128,128,3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(64,3,strides=3,activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(units = 128,activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(units = 120, activation='softmax'))
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

#generate network
model = Sequential()
model.add(Conv2D(32,3,strides=3,input_shape=(128,128,3),activation='relu',data_format = "channels_last"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,3,strides=3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,3,strides=3,activation='relu'))


model.add(Lambda(global_average_pooling, output_shape=global_average_pooling_shape))

#model.add(Dense(units = 128,activation='relu'))
#model.add(Dropout(0.1))

model.add(Dense(120, activation = 'softmax', init='uniform'))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
#model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())


#from keras.models import model_from_json
#json_file = open('model_vgg16.json','r')
#nn_json = json_file.read()
#json_file.close()
#model = model_from_json(nn_json)
#weights_file = "weights_vgg16.hdf5"
#model.load_weights(weights_file)
#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('train',
                                                  target_size=(128, 128),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle = True)

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical',                                         
                                            shuffle = True)

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    './base.model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=30,
    verbose=1,
    mode='auto'
)
tensorboard = TensorBoard(
    log_dir = './logs',
    histogram_freq=0,
    batch_size=16,
    write_graph=True,
    write_grads=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename= "training_csv.log",
    separator = ",",
    append = False
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1, 
    mode='auto'
)

callbacks = [checkpoint,tensorboard,csvlogger,reduce]

history = model.fit_generator(training_set,
                    steps_per_epoch=12000//32,
                    epochs=50,
                    validation_data=test_set,
                    callbacks = callbacks,
                    validation_steps=8580//32)



#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix

#Compute probabilities
test_set_size = 8580
batch_size = 32
Y_pred = model.predict_generator(test_set,steps=test_set_size//batch_size+1)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
target_names = training_set.class_indices.keys()
series = pd.Series(target_names)
series.to_csv('target_names_categorical_dog.csv')

#classification report
class_rep = classification_report(test_set.classes,y_pred)
#write to test file
text_file = open("classification_report_categorical_dog.txt", "w")
text_file.write(class_rep)
text_file.close()

#confusion matrix
conf_matrix = confusion_matrix(test_set.classes,y_pred)
conf_matrix = pd.DataFrame(conf_matrix)
conf_matrix.to_csv('conf_matrix_categorical_dog.csv')


from keras.models import model_from_json
nn_json = model.to_json()
with open('model_vgg16.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights_vgg16.hdf5"
model.save_weights(weights_file,overwrite=True)
