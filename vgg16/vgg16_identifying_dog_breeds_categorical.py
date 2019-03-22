import pandas as pd
#from keras.models import Sequential
#from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,GlobalAveragePooling2D,Dropout
#from keras.applications.vgg16 import VGG16


from keras.models import model_from_json
json_file = open('model_vgg16.json','r')
nn_json = json_file.read()
#json_file.close()
model = model_from_json(nn_json)
weights_file = "weights_vgg16.hdf5"
model.load_weights(weights_file)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

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
                    steps_per_epoch=12000,
                    epochs=25,
                    validation_data=test_set,
                    callbacks = callbacks,
                    validation_steps=8580)

from keras.models import model_from_json
nn_json = model.to_json()
with open('model_vgg16_2.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights_vgg16_2.hdf5"
model.save_weights(weights_file,overwrite=True)

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

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
