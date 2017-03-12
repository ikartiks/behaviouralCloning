import csv,cv2,numpy
import os
from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

import matplotlib.pyplot as plt

#a=numpy.array([[1,2,3], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
#b=numpy.array([[4, 5, 6], [7, 8, 9]])
#x=numpy.append(a,b , axis=0)
#print(x, a.shape,' ',b.shape)

#a=[1,2,4,5,6,7,8,9,10,0]
#offset=3
#batch_size=3
#batch_samples = a[offset:offset+batch_size]


print('Done')

#Writing same code using generators since this takes a lot of memory

#lines=[]
#images=[] #input
#measurements=[] #y


#for local path
#localPathBase='/Users/kartikshah/Documents/selfDrivingCarProjects/CarND-Behavioral-Cloning-P3/data/IMG/'

#for relative path (should work on ec2 instance too)
localPathBase='data/IMG/'

correction=0.1

samples = []
with open('data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if(row[3]=='steering'):
            continue
        samples.append(row)
        
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = numpy.array(image1, dtype = numpy.float64)
    random_bright = .5+numpy.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = numpy.array(image1, dtype = numpy.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*numpy.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*numpy.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = numpy.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = numpy.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*numpy.random.uniform()
    if numpy.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if numpy.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):#range(start, stop, step)
            #return data from offset to offset+batchs_size 0-32,32-64...
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                #getting the angles
                center_angle = float(batch_sample[3])
                left_angle  = center_angle + correction
                right_angle = center_angle - correction
                
                #take 3 images
                center_image = cv2.imread(localPathBase+batch_sample[0].split('/')[-1])
                #center_image = cv2.imread(batch_sample[0])
                images.append(center_image)
                angles.append(center_angle)
                
                
                
                right_image = cv2.imread(localPathBase+batch_sample[2].split('/')[-1])
                #right_image = cv2.imread(batch_sample[2])
                images.append(right_image)
                angles.append(right_angle)
                
                left_image = cv2.imread(localPathBase+batch_sample[1].split('/')[-1])
                #left_image = cv2.imread(batch_sample[1])
                images.append(left_image)
                angles.append(left_angle)
                
                
                #add flipped images
                
                image_flipped = numpy.fliplr(left_image)
                measurement_flipped = -left_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)
                
                image_flipped = numpy.fliplr(center_image)
                measurement_flipped = -center_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)
                
                image_flipped = numpy.fliplr(right_image)
                measurement_flipped = -right_angle
                images.append(image_flipped)
                angles.append(measurement_flipped)
               
                
                
                
                #add brightned images
                
                bright_image = augment_brightness_camera_images(center_image)
                images.append(bright_image)
                angles.append(center_angle)
                
                bright_image = augment_brightness_camera_images(left_image)
                images.append(bright_image)
                angles.append(left_angle)
                
                bright_image = augment_brightness_camera_images(right_image)
                images.append(bright_image)
                angles.append(right_angle)
                
                #add random shadow to 3 images
                images.append(add_random_shadow(center_image))
                angles.append(center_angle)
                
                images.append(add_random_shadow(right_image))
                angles.append(center_angle)
                
                images.append(add_random_shadow(left_image))
                angles.append(center_angle)
                
                
                
                

            # trim image to only see section with road
            X_train = numpy.array(images)
            y_train = numpy.array(angles)
            #print(type(X_train[0]),X_train[0].shape)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('generators generated')

#print (x_train[0].shape,'  ',x_train[1].shape)
from keras.models import Sequential
from keras.layers import Dense, Flatten,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
print('Started')
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
#model.fit(x_train, y_train, nb_epoch=3, validation_split=0.2,shuffle=True,verbose=1)
#multiply samples per epoch for both by 3 since for every sample we are takimg center,left,right images
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*12,\
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*12,\
                    nb_epoch=3,verbose=1)

model.save('model-10.h5')


print('model saved')