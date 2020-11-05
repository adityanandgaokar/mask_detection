
import os
from imutils import paths
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


image_files = list(paths.list_images('D:/Projects/Mask_detection/dataset'))

data = []

labels = []

batch_size = 32

number_of_epochs = 20


##picture = load_img('D:/Projects/Mask_detection/dataset/with_mask/aditya1.jpg', target_size=(224,224))


for path in image_files:

    label = path.split(os.path.sep)[-2]
    
    picture = load_img(path, target_size=(224,224))

    picture = img_to_array(picture)

    picture = preprocess_input(picture)

    data.append(picture)
    labels.append(label)
    

data = np.array(data, dtype= 'float32')
labels = np.array(labels)

le = LabelBinarizer()

labels = le.fit_transform(labels)
labels = to_categorical(labels)


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size = 0.2, stratify = labels,
                                                    random_state= 42)

print(x_train.shape)
print(y_train.shape)
        
augmentation = ImageDataGenerator(rotation_range =20,
                                  zoom_range = 0.15,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range=0.15,
                                  horizontal_flip =True,
                                  fill_mode ='nearest')


## model structure

model_base = MobileNet(weights ='imagenet', include_top=False,
                         input_tensor = Input(shape = (224,224,3)))



model_head = model_base.output

##model_head = Input(shape = (224,224,3))
model_head = AveragePooling2D(pool_size=(7,7))(model_head)
model_head = Flatten(name='flatten')(model_head)
model_head = Dense(128, activation = 'relu')(model_head)
model_head = Dropout(0.5)(model_head)
model_head = Dense(2, activation='softmax')(model_head)

model = Model(inputs = model_base.input, outputs= model_head)

for layer in model_base.layers :
    layer.trainable = False


opt = Adam(lr = 1e-4, decay=1e-4/ number_of_epochs)

model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

model_fit = model.fit(
    augmentation.flow(x_train, y_train, batch_size = batch_size),
    steps_per_epoch= int(len(x_train)/batch_size),
    validation_data = (x_test, y_test),
    validation_steps = int(len(x_test)/ batch_size),
    epochs = number_of_epochs)

model.save('D:/Projects/Mask_detection/Models/mask_detection.h5')


