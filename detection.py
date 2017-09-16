
#Neural network for face key dots detection 
#(Guliev, date: 25.05.16)

def set_backend():
    import os
    os.environ['KERAS_BACKEND'] = 'theano'
    from keras.backend import set_image_dim_ordering
    set_image_dim_ordering('th')
    
set_backend()

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Dropout,Convolution2D, ZeroPadding2D
from skimage.io import imread, imsave
from skimage.transform import resize
from numpy import zeros
from os import listdir
from sklearn import preprocessing
from skimage.transform import rotate

def I(image):
    if len(image.shape) == 3:
        return image[:,:,0]*0.299+image[:,:,1]*0.587+image[:,:,2]*0.114
    else:
        return image

def train_detector(train_gt, train_img_dir, fast_train):
    size = 64
    #train_img = zeros((len(train_gt),size,size,3), dtype=int)
    train_img = zeros((len(train_gt),1,size,size), dtype=float)
    coords_arr = zeros((len(train_gt),28), dtype=float)
    
    c = 0
    for filename, coords in train_gt.items():
        print(filename)
        img = I(imread(train_img_dir + '/' + filename))
        img_r = resize(img, (size, size))
        #img_r = preprocessing.normalize(img_r)
        #img_r[:,:,0] = preprocessing.normalize(img_r[:,:,0])
        #img_r[:,:,1] = preprocessing.normalize(img_r[:,:,1])
        #img_r[:,:,2] = preprocessing.normalize(img_r[:,:,2])
        train_img[c][0,:,:] = img_r
        coords_arr[c] = coords * size / img.shape[0]
        c+=1
    
    model = Sequential()

    model.add(Conv2D(8, (7, 7), input_shape=(1,size, size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (4, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(18*32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(12*32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(28))
    
    model.compile(loss='mean_squared_error',
              optimizer='Adadelta',
              metrics=['accuracy'])
    if fast_train is True:
        model.fit(train_img, coords_arr, epochs=30, batch_size=size)
    else:
        model.fit(train_img, coords_arr, epochs=5, batch_size=size)
    return model



def detect(model, test_img_dir):
    size = 64
    res = {}
    files = listdir(test_img_dir)
    test_img = zeros((len(files),1,size,size), dtype=int)
    
    c = 0
    for filename in files[0:len(files)-1]:
        print(filename)
        img = I(imread(test_img_dir + '/' + filename))
        res[filename] = img.shape[0]/size
        img_r = resize(img, (size, size))
        #img_r = preprocessing.normalize(img_r)
        #img_r[:,:,0] = preprocessing.normalize(img_r[:,:,0])
        #img_r[:,:,1] = preprocessing.normalize(img_r[:,:,1])
        #img_r[:,:,2] = preprocessing.normalize(img_r[:,:,2])
        test_img[c][0,:,:] = img_r
        c+=1
        
    out = model.predict(test_img, batch_size=100)
    
    c = 0
    for filename in files[0:len(files)-1]:
        #print("out ", out[c])
        res[filename] *= out[c]
        print(res[filename][:7])
        print(res[filename][7:14])
        print(res[filename][14:21])
        print(res[filename][21:])
        print('\n')
        c+=1
    return res

def makeit(model, train_gt, train_img_dir):
    size = 64
    train_img = zeros((len(train_gt),1,size,size), dtype=float)
    coords_arr = zeros((len(train_gt),28), dtype=float)
    
    c = 0
    for filename, coords in train_gt.items():
        print(filename)
        img = I(imread(train_img_dir + '/' + filename))
        #img_r = preprocessing.normalize(img_r)
        #img_r[:,:,0] = preprocessing.normalize(img_r[:,:,0])
        #img_r[:,:,1] = preprocessing.normalize(img_r[:,:,1])
        #img_r[:,:,2] = preprocessing.normalize(img_r[:,:,2])
        #if filename == "000000.jpg":
        #    print(coords * size/img.shape[0])
            
            #imsave(filename, img)
        #if filename == "000014.jpg":
        #    print(coords * size/img.shape[0])
            
            #imsave(filename, img)
        train_img[c][0,:,:] = resize(img, (size, size))
        coords_arr[c] = coords * size / img.shape[0]
        c+=1
    i = 0
    for i in range(19):
        print(i+24)
        #es=EarlyStopping(monitor='val_loss',patience=3, verbose=0)
        model.fit(train_img, coords_arr, epochs=5, batch_size=size)#, verbose=1,callbacks=[checkpoints,es])
        model.save('facepoints_model' + str(i+24) + '.hdf5')
    return model




