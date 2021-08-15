## Intro

...

## Empirical analysis

First step in the analysis is downloading the dataset and importing neccesary packages:

``` {.python}
# import of nessesary packages
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Input, backend, Model, optimizers, layers

# cloning the repository of the project with the dataset
# ! git clone https://kmatusz:xxxxxxxxxxx@github.com/kmatusz/cell_segmentation.git
! git clone -b bez_babli https://kmatusz:xxxxxxxxxxx@github.com/kmatusz/cell_segmentation.git
# ! rm cell_segmentation/ -r

# set environment depending on colab vs. local running
path_to_main = 'cell_segmentation/' # For running on google colab
# path_to_main = '' # For local running
```

### Data preparation

The dataset provided shows a one 24-hour image capture of the experiment.

![Video from the experiment - input for the model](in.gif)

On the original movie there are 16 "pools" - in each pool there are one or two cells. Couple of challenges had to be addresed by us to provide clean dataset for the modeling.

Firstly, after cropping one picture dataset into 16 smaller ones, on each photo there was still some background. We tried running the network on such dataset, however after first run of the network it became clear that the model is very prone to misclassification. The network learned how to classify pool vs. background, which made sense as the black corners take a large part of the picture. That is why we flood-filled the background with gray color - specifically color indicated by mean value of each pixel in the dataset. Such simple method as flood fill did not work so well on each case, but it mitigated the problem with misclassification.

![Data preparation process](data_preparation.png)

Secondly, we had to manually annotate the dataset. We have selected every 15th frame to annotate, which gave 10 annotated images per one cell (160 in total). We have created the annotation as a separate picture, on which black pixels equal mean the cell, while white ones-background. A standard format for mask in image segmentation tasks is often a json file with shapes specification. However, the cells sometimes had very irregular shapes, and thus the professional annotation software did not suit our needs.

Because of a very small datset, we have decided to do simple train/validation split - without a separate, third test set. Simple strategies for the split like randomly splitting the images into 2 sets would not work in our case, as data leakage would occur. That is why train set consisted of first 11 pools, while test set - 5 remaining ones.

A standard procedure in dealing with image recognition is data augmentation. This procedure becomes of the highest importance for the model performance when only couple hundred samples are provided. That is why we have used the following data augmentations and preprocessing on the train set:

-   rotation
-   shift in both width and height
-   zoom
-   horizontal flip
-   rescale values of pixels from [1,255] to [0,1]

On validation set we have only rescaled values of pixels to desired range.

Function for loading images and masks from folder - compatible type with keras. Change size of images to grayscale, change shape of numpy array:

``` {.python}
def load_images(path = 'data/keras_input/train/'):
    images_names = [x for x in os.listdir(path) if 'x' in x]
    masks_names = [x for x in os.listdir(path) if 'y' in x]
    images_names.sort()
    masks_names.sort()
    
    images_list = []
    masks_list = []
    for i in range(len(images_names)):
        img = image.load_img(os.path.join(path, images_names[i]), target_size=(256, 256),color_mode='grayscale')
        img_mask = image.load_img(os.path.join(path, masks_names[i]), target_size=(256, 256),color_mode='grayscale')
        images_list.append(img)
        masks_list.append(img_mask)
        
    images_arrays_list = [image.img_to_array(img) for img in images_list]
    images_arrays_list = [img_array.reshape((1,) + img_array.shape) for img_array in images_arrays_list]
    images_np = np.concatenate(images_arrays_list)
    
    masks_arrays_list = [image.img_to_array(img) for img in masks_list]
    masks_arrays_list = [img_array.reshape((1,) + img_array.shape) for img_array in masks_arrays_list]
    masks_np = np.concatenate(masks_arrays_list)
    return (images_np, masks_np)
```

Create Keras data generators. Besides resizing the image it performs data augmentation on training dataset.

``` {.python}
datagen = image.ImageDataGenerator(
    rotation_range=180, # In degrees, how much to rotate the image
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2, # Spłaszczenie
    zoom_range=0.01, # Zoom +/-
    horizontal_flip=True, 
    rescale = 1./255,
    fill_mode='nearest' # How to fill newly created pixels,

)

datagen_val = image.ImageDataGenerator(
    rescale = 1./255
)
```

Function for creating flow (python generator) of data to keras.

``` {.python}
def create_data_flow(datagen, images_np, masks_np, batch_size = 1, seed = 2):
    img_flow = datagen.flow(images_np, batch_size=batch_size,seed=seed)
    mask_flow = datagen.flow(masks_np, batch_size=batch_size,seed=seed)
    return zip(img_flow, mask_flow)
```

``` {.python}
# Train
images_np, masks_np = load_images(path_to_main + 'data2/train/')
flow_train = create_data_flow(datagen, images_np, masks_np)
```

``` {.python}
# Validation 
images_np_val, masks_np_val = load_images(path_to_main + 'data2/validation/')
flow_validation = create_data_flow(datagen_val, images_np_val, masks_np_val, batch_size = 16)
```

Sanity checks - if the images and masks are of the same kind:

``` {.python}
a,b = next(flow_train)
```

``` {.python}
Image.fromarray((a[0,:,:,0]*255).astype(np.uint8),'P')
```

![Original image - preprocessed by data augmentation](output_13_0.png)

``` {.python}
Image.fromarray((b[0,:,:,0]*255).astype(np.uint8),'P')
```

![Mask - also preprocessed by data augmentation](output_14_0.png)

### Modeling

A state-of-the art approach to segmentation task is U-Net deep learning network architecture. It was proposed in 2013 by (...).

Our first approach was to replicate that model on our dataset. We have decided on this model for a couple of reasons:

1. This model is widely-used for all tasks of segmentation and serves as builiding block for more modern segmentation approaches. 
2. In the original paper the segmentation is done in the same domain, that is segmentation of microscope imagery of a cell. 
3. The authors had to address the same issue that we had, namely lack of large, annotated dataset for the training. This is somehow a standard in biological imagery, as each problem is very specific. Besides that, transfer learning in such setting is not as easy as in standard clssification models, as it would require direct manipulation on each network layer using tensorflow functional API.

The U-Net model as specified in original paper consists of 2 parts - contracting path and expansive path. Contracting path resembles a typical convolutional neural network. It consists of couple convolutional blocks. Each block has two convolutional layers, followed by max-pooling layer and dropout. In each consecutive block, number of feature channels is doubled, while the size of the output is divided by 2 (because of max pooling).

Second part of the network is expansive path. In each block, 2 things are happening. First, Transpose layer is applied on input from previous block in expansive path. This is meant to upsample the input, as at the end of the network output is expected to have the size of the original image. Second, output is concatenated with input from the contracting path, and 2 convolution layers are applied.

The purpose of the contracting path is feature extraction from the image, without precise, pixel-wise information about the location of particular feature. This precision is obtained from the expansive path.

Following the original paper, we have included 4 blocks in contracting path. We have used 0,25 dropout in the first block, and 0.5 in later ones. In all middle layers, we have used a relu activation function. In the last convolution layer sigmoid activation is used, as desired output is probability of a pixel being a blastocyst vs background. In the first contracting block 32 feature channels are used, and numer of channels is doubled in each block.

We have monitored loss metric on train and validation datasets. For each epoch, 1000 examples generated using data augmentation are fed into the network. In the first approach simple RMSprop optimizer is used with default learning rate.

``` {.python}
img_size_target = (256)
start_neurons = 32

# convolution blocks - conv -> conv -> pooling -> dropout
input_layer = Input((img_size_target, img_size_target, 1))

conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
pool1 = layers.Dropout(0.25)(pool1)

conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
pool2 = layers.Dropout(0.5)(pool2)

conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = layers.MaxPooling2D((2, 2))(conv3)
pool3 = layers.Dropout(0.5)(pool3)

conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = layers.MaxPooling2D((2, 2))(conv4)
pool4 = layers.Dropout(0.5)(pool4)


# Middle layers - conv -> conv (no pooling, no dropout)
convm = 3
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

# Final layers
# 1. Get output from middle of conv part
# 2. Conv2DTranspose 
# 3. concatenate with convolution layer output
# 4. Dropout
# 5. Conv
# 6. conv
deconv4= layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = layers.concatenate([deconv4, conv4])

uconv4 = layers.Dropout(0.5)(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

deconv3= layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = layers.concatenate([deconv3, conv3])
uconv3 = layers.Dropout(0.5)(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

deconv2= layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = layers.concatenate([deconv2, conv2])
uconv2 = layers.Dropout(0.5)(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

deconv1= layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = layers.concatenate([deconv1, conv1])
uconv1 = layers.Dropout(0.5)(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
```

``` {.python}
model2 = Model(input_layer, output_layer)
```

``` {.python}
model2.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 256, 256, 1) 0                                            
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, 256, 256, 32) 320         input_2[0][0]                    
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, 256, 256, 32) 9248        conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)  (None, 128, 128, 32) 0           conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    dropout_8 (Dropout)             (None, 128, 128, 32) 0           max_pooling2d_4[0][0]            
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, 128, 128, 64) 18496       dropout_8[0][0]                  
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, 128, 128, 64) 36928       conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)  (None, 64, 64, 64)   0           conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    dropout_9 (Dropout)             (None, 64, 64, 64)   0           max_pooling2d_5[0][0]            
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, 64, 64, 128)  73856       dropout_9[0][0]                  
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, 64, 64, 128)  147584      conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_6 (MaxPooling2D)  (None, 32, 32, 128)  0           conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    dropout_10 (Dropout)            (None, 32, 32, 128)  0           max_pooling2d_6[0][0]            
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, 32, 32, 256)  295168      dropout_10[0][0]                 
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, 32, 32, 256)  590080      conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    max_pooling2d_7 (MaxPooling2D)  (None, 16, 16, 256)  0           conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    dropout_11 (Dropout)            (None, 16, 16, 256)  0           max_pooling2d_7[0][0]            
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, 16, 16, 512)  1180160     dropout_11[0][0]                 
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, 16, 16, 512)  2359808     conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_4 (Conv2DTrans (None, 32, 32, 256)  1179904     conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    concatenate_4 (Concatenate)     (None, 32, 32, 512)  0           conv2d_transpose_4[0][0]         
                                                                     conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    dropout_12 (Dropout)            (None, 32, 32, 512)  0           concatenate_4[0][0]              
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, 32, 32, 256)  1179904     dropout_12[0][0]                 
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, 32, 32, 256)  590080      conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_5 (Conv2DTrans (None, 64, 64, 128)  295040      conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    concatenate_5 (Concatenate)     (None, 64, 64, 256)  0           conv2d_transpose_5[0][0]         
                                                                     conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    dropout_13 (Dropout)            (None, 64, 64, 256)  0           concatenate_5[0][0]              
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, 64, 64, 128)  295040      dropout_13[0][0]                 
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, 64, 64, 128)  147584      conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_6 (Conv2DTrans (None, 128, 128, 64) 73792       conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 128, 128, 128 0           conv2d_transpose_6[0][0]         
                                                                     conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    dropout_14 (Dropout)            (None, 128, 128, 128 0           concatenate_6[0][0]              
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, 128, 128, 64) 73792       dropout_14[0][0]                 
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, 128, 128, 64) 36928       conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    conv2d_transpose_7 (Conv2DTrans (None, 256, 256, 32) 18464       conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    concatenate_7 (Concatenate)     (None, 256, 256, 64) 0           conv2d_transpose_7[0][0]         
                                                                     conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    dropout_15 (Dropout)            (None, 256, 256, 64) 0           concatenate_7[0][0]              
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, 256, 256, 32) 18464       dropout_15[0][0]                 
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, 256, 256, 32) 9248        conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, 256, 256, 1)  33          conv2d_36[0][0]                  
    ==================================================================================================
    Total params: 8,629,921
    Trainable params: 8,629,921
    Non-trainable params: 0
    __________________________________________________________________________________________________

``` {.python}
# Callbacks
from tensorflow.keras import callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss',
                                  patience=5,),
    callbacks.ModelCheckpoint(filepath=path_to_main + 'models/model_bable.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
# model.fit(x, y,epochs=10,batch_size=32,,validation_data=(x_val, y_val))
```

``` {.python}
model2.compile(optimizer="rmsprop", loss="binary_crossentropy")

# Train the model, doing validation at the end of each epoch.
epochs = 20
steps_per_epoch = 1000
history = model2.fit(flow_train, validation_data=flow_validation, 
                     validation_steps = 1,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     callbacks=callbacks_list)
```

    Epoch 1/20
    1000/1000 [==============================] - 46s 36ms/step - loss: 857.5335 - val_loss: 0.1208
    [...]
    Epoch 14/20
    1000/1000 [==============================] - 34s 34ms/step - loss: 0.1128 - val_loss: 0.3663

``` {.python}
import matplotlib.pyplot as plt

loss = history.history['loss'][1:]
val_loss = history.history['val_loss'][1:]

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

![](output_22_0.png)

The model seems to not improve during the training. From other experiments we know that a loss at around 0.08 is the lowest as we could get. This means that we have gotten close to the global minimum very fast - just after one epoch. Also, below we have shown a predicted image - it is clear that the performance could be way better, but the net learns to recognize the cell correctly.

``` {.python}
a, b = next(flow_validation)
preds = model2.predict(a)

plt.imshow(a[0,:,:,0]*255, cmap='gray')
plt.imshow(preds[0,:,:,0]*255, cmap='jet', alpha=0.5)
```

![](output_24_1.png)

A good thing is that there is no overfitting visible - sometimes the loss on the test set is even lower than on training. This is a strange, but desired situation - one possible explanation of such behaviour is usage of dropout.

Such fast convergence to a minimum is not necessarily desired. It is possible that the network is stuck in a local minimum after first few epochs and cannot leave it in search for global one. One solution for too fast convergence is simply lowering the learning rate of the optimizer. This is what we have tried in our next attempt - we have lowered learinig rate of RMSprop from default 0.0001 to 0.00001. Other step we took to slowen the learning was to lower the number of samples included in each epoch - from 2000 to 200:


``` {.python}
img_size_target = (256)
start_neurons = 32
```

``` {.python}
# convolution blocks - conv -> conv -> pooling -> dropout
input_layer = Input((img_size_target, img_size_target, 1))

conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
pool1 = layers.Dropout(0.25)(pool1)

conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
pool2 = layers.Dropout(0.5)(pool2)

conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = layers.MaxPooling2D((2, 2))(conv3)
pool3 = layers.Dropout(0.5)(pool3)

conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = layers.MaxPooling2D((2, 2))(conv4)
pool4 = layers.Dropout(0.5)(pool4)


# Middle layers - conv -> conv (no pooling, no dropout)
convm = 3
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

# Final layers
# 1. Get output from middle of conv part
# 2. Conv2DTranspose 
# 3. concatenate with convolution layer output
# 4. Dropout
# 5. Conv
# 6. conv
deconv4= layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = layers.concatenate([deconv4, conv4])

uconv4 = layers.Dropout(0.5)(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

deconv3= layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = layers.concatenate([deconv3, conv3])
uconv3 = layers.Dropout(0.5)(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

deconv2= layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = layers.concatenate([deconv2, conv2])
uconv2 = layers.Dropout(0.5)(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

deconv1= layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = layers.concatenate([deconv1, conv1])
uconv1 = layers.Dropout(0.5)(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
```

``` {.python}
# Callbacks
from tensorflow.keras import callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='loss',
                                  patience=10,),
    callbacks.ModelCheckpoint(filepath=path_to_main + 'models/model3.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
# model.fit(x, y,epochs=10,batch_size=32,,validation_data=(x_val, y_val))
```

``` {.python}
model3 = Model(input_layer, output_layer)
model3.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.00001))
# Train the model, doing validation at the end of each epoch.
epochs = 100
steps_per_epoch = 200
history = model3.fit(flow_train, validation_data=flow_validation, 
                     validation_steps = 1,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     callbacks=callbacks_list)
```

    Epoch 1/100
    200/200 [==============================] - 10s 37ms/step - loss: 0.6374 - val_loss: 0.5731
    [...]
    Epoch 80/100
    200/200 [==============================] - 7s 35ms/step - loss: 0.0881 - val_loss: 0.5747

``` {.python}
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

![](output_32_0.png)

It is clear that the model started overfitting after 20-30 epochs. After that time validation loss skyrockets. This is an undesired behaviour. However, it can be seen that even trainig loss is not improving much after 15-20 epochs. This is a sign to stop the model training around that time. Anyway, the loss achieved is not astounding - the minimum on the validation set is around 0.16, while in the initial model it was twice as low - 0.08.


In all previous tries we have included dropout extensively. This type of layer was however not used in the original U-Net paper, possibly because the paper describing dropout layer was presented just one year earlier, in 2014. Anyway, we have tried removing dropout layer from the previous model. The rest of the parameters stayed the same as in the previous model.

``` {.python}
img_size_target = (256)
start_neurons = 32
```

``` {.python}
# convolution blocks - conv -> conv -> pooling -> dropout
input_layer = Input((img_size_target, img_size_target, 1))

conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = layers.MaxPooling2D((2, 2))(conv1)


conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = layers.MaxPooling2D((2, 2))(conv2)


conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = layers.MaxPooling2D((2, 2))(conv3)


conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = layers.MaxPooling2D((2, 2))(conv4)


# Middle layers - conv -> conv (no pooling, no dropout)
convm = 3
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

# Final layers
# 1. Get output from middle of conv part
# 2. Conv2DTranspose 
# 3. concatenate with convolution layer output
# 4. Dropout
# 5. Conv
# 6. conv
deconv4= layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = layers.concatenate([deconv4, conv4])

uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

deconv3= layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = layers.concatenate([deconv3, conv3])

uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

deconv2= layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = layers.concatenate([deconv2, conv2])

uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

deconv1= layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = layers.concatenate([deconv1, conv1])

uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
```

``` {.python}
# Callbacks
from tensorflow.keras import callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='loss',
                                  patience=10,),
    callbacks.ModelCheckpoint(filepath=path_to_main + 'models/model4.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
# model.fit(x, y,epochs=10,batch_size=32,,validation_data=(x_val, y_val))
```

``` {.python}
model4 = Model(input_layer, output_layer)
model4.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.000001))
# Train the model, doing validation at the end of each epoch.
epochs = 100
steps_per_epoch = 200
history = model4.fit(flow_train, validation_data=flow_validation, 
                     validation_steps = 1,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     callbacks=callbacks_list)
```

    Epoch 1/100
    200/200 [==============================] - 9s 34ms/step - loss: 0.6881 - val_loss: 0.6859
    [...]
    Epoch 98/100
    200/200 [==============================] - 7s 33ms/step - loss: 0.1485 - val_loss: 0.1555

``` {.python}
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

![](output_40_0.png)

The obtained results are very surprising. The dropout layer is usually included in the model to reduce the overfitting. In our case, when comparing results with and without dropout, resulting best validation loss is roughly the same (0.161 vs. 0.155). Two things are however different. In dropout case, the convergence to minimum appeared way faster (20 epochs, in with no dropout case - around 60). Another finding is that in dropout case heavier overfitting was visible - especially after 30 epochs the validation loss was way higher than on training dataset. But the difference between the sets is clearly visible from the beginning. In no dropout case losses for the two sets are very close to each other.


Despite the fact that first model has learned way too quickly, and improvement during the training practically didn't occur, the results on the test set were the best from all models tested. However, strategy of stopping the model training just after it gets the loss at around 0.08 should be applied. The results from this model epoch-by-epoch were mostly at random, that is why we have decided to run this model but stop it by hand when the loss achieves desired level.

``` {.python}
img_size_target = (256)
start_neurons = 32

# convolution blocks - conv -> conv -> pooling -> dropout
input_layer = Input((img_size_target, img_size_target, 1))

conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = layers.MaxPooling2D((2, 2))(conv1)
pool1 = layers.Dropout(0.25)(pool1)

conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = layers.MaxPooling2D((2, 2))(conv2)
pool2 = layers.Dropout(0.5)(pool2)

conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = layers.MaxPooling2D((2, 2))(conv3)
pool3 = layers.Dropout(0.5)(pool3)

conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = layers.MaxPooling2D((2, 2))(conv4)
pool4 = layers.Dropout(0.5)(pool4)


# Middle layers - conv -> conv (no pooling, no dropout)
convm = 3
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

# Final layers
# 1. Get output from middle of conv part
# 2. Conv2DTranspose 
# 3. concatenate with convolution layer output
# 4. Dropout
# 5. Conv
# 6. conv
deconv4= layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = layers.concatenate([deconv4, conv4])

uconv4 = layers.Dropout(0.5)(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

deconv3= layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = layers.concatenate([deconv3, conv3])
uconv3 = layers.Dropout(0.5)(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

deconv2= layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = layers.concatenate([deconv2, conv2])
uconv2 = layers.Dropout(0.5)(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

deconv1= layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = layers.concatenate([deconv1, conv1])
uconv1 = layers.Dropout(0.5)(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

output_layer = layers.Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
```

``` {.python}
model5 = Model(input_layer, output_layer)
```

``` {.python}
# Callbacks
from tensorflow.keras import callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss',
                                  patience=2,),
    callbacks.ModelCheckpoint(filepath=path_to_main + 'models/model5.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
# model.fit(x, y,epochs=10,batch_size=32,,validation_data=(x_val, y_val))
```

``` {.python}
model5.compile(optimizer="rmsprop", loss="binary_crossentropy")

# Train the model, doing validation at the end of each epoch.
epochs = 20
steps_per_epoch = 1000
history = model5.fit(flow_train, validation_data=flow_validation, 
                     validation_steps = 1,
                     epochs=epochs,
                     steps_per_epoch=steps_per_epoch,
                     callbacks=callbacks_list)
```

    Epoch 1/20
    1000/1000 [==============================] - 37s 35ms/step - loss: 0.9134 - val_loss: 0.1734
    Epoch 2/20
    1000/1000 [==============================] - 34s 34ms/step - loss: 0.1755 - val_loss: 0.0941
    Epoch 3/20
     127/1000 [==>...........................] - ETA: 29s - loss: 0.1248


### Visual evaluation of the model

Next step of the analysis is actual prediction on the test dataset. Final prediction from the model should be checked visually, as it will be actually used by the research team. In this section, we have started with showing the output on the single image. Later, we have used the model on the movie from the experiment, as it will be used in the wild. We have used a thresholding at 0.8 (determined manually).

The ultimate goal of the model we have created was to estimate the changes in blastocyst size over time. That is why we have compared predicted area of the cell with a human-generated manual estimation of the size.

``` {.python}
# Validation 
images_np_val, masks_np_val = load_images(path_to_main + 'data2/validation')
flow_validation = create_data_flow(datagen_val, images_np_val, masks_np_val, batch_size = 16)
```

``` {.python}
a, b = next(flow_validation)
preds = model5.predict(a)
Image.fromarray((a[0,:,:,0]*255).astype(np.uint8),'P')
```

![Original image](output_53_0.png)

``` {.python}
Image.fromarray((b[0,:,:,0]*255).astype(np.uint8),'P')
```

![Original mask](output_55_0.png)

``` {.python}
Image.fromarray((preds[0,:,:,0]*255).astype(np.uint8),'P')
```

![Predicted mask](output_57_0.png)

``` {.python}
plt.imshow(a[0,:,:,0]*255, cmap='gray')
plt.imshow(preds[0,:,:,0]*255, cmap='jet', alpha=0.5)
```

![Original image with predicted mask](output_61_1.png)

``` {.python}
plt.imshow(b[0,:,:,0]*255, cmap='gray')
plt.imshow(preds[0,:,:,0]*255, cmap='jet', alpha=0.5)
```

![Masks - original and predicted](output_63_1.png)

Next stage of visual output inspection is plotting the prediction directly on the input movie. It can be seen that the model is performing almost perfectly on the first half of the movie. In the later part the prediction is also very good, however the model has problems with correctly classifying the bubbles that pop out of the main cell. We have tried to remove that problem by adding to the train set more images from that later part, but this did not improve the prediction. Instead, the model became more sensitive for the random impurities in the pool (like the ones in the bottom part of the picture). That is why after the consultation with the research team we have opted for previous solution.

![Final model prediction](out5.gif)

Final step of output examination is testing whether it is possible to estimate size of the cell correctly. We obtained manual recordings of the cell size for one example cell (shown on the GIF above). We have compared this measurement with output of the model. Below the plot comparing the measurements is presented. The blue line is manual measurement, while orange one - measurement from the model. These two lay very close to each other. The reason for small disrepancies at around 80th frame are the bubbles we have mentioned before.

``` {.python}
import matplotlib.pyplot as plt
def normalize(x):
    return (x-x.min())/(x.max()-x.min())

excel_0 = [5466,
5466,
...]

plt.plot(normalize(np.array(area_filled)))
plt.plot(normalize(np.array(excel_0)))
```

![Comparison of size estimation - manual and predicted by the model](output_71_1_labs_dodane.png)


## Summary

The main goal of this study was ... (coś z intro przekleić)

A basis for all of our models was U-Net architecture. After data preparation, we have experimented with optimizer learning rate, usage of dropout and early stopping rules for the model. The model with very big learning rate and large number of iterations per each epoch turned out to produce the best resutls. However, due to fast learning the results were very unstable, so manual stopping of the training at proper time was needed. 

Models with smaller learning rate showed "typical" properties of deep learning model, that is slow descent to a objective function minimum. In one such model we have incuded standard dropout layers. These layers, although typically intended for fighting overfitting, were not helpful in this case - these actually introduced overfitting to the training data. What it actually helped in was speed of convergence - model with dropout converge to a mininmum roughlt 2 times faster than the other one. 

After successful training of the model we were able to use it in real-world case - that is for measuring the size of a cell in an 24h in-vitro experiment. The results of this estimation were very close to the previously, manually gathered measurements. Our work will facilitate the day-to-day experiments, as a very time-consuming process of manual cells annotation will no longer be a burden.























