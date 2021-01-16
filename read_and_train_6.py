import csv
import cv2
import numpy as np

dataPath = 'data'
driving_log_list = ['driving_log.csv', 'driving_log_2.csv']

correction = 0.5 # this is a parameter to tune

def get_image_from_sourcepath(source_path):
    filename = source_path.split('/')[-1]
    current_path = './{}IMG/{}'.format(dataPath,filename)
    image = cv2.imread(current_path)
    return image

def read_lines_from_filename(filename):
    local_lines = []
    with open('./{}/{}'.format(dataPath, filename)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            local_lines.append(line)
    return local_lines

lines = []
print('Reading from: ./{}/'.format(dataPath))
for d_log in driving_log_list:
    print('Reading file: {}'.format(dataPath))
    lines_readed = read_lines_from_filename(d_log)
    lines.extend(lines_readed)

images = []
measurements = []

for line in lines:
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    image_center = get_image_from_sourcepath(line[0])
    image_left = get_image_from_sourcepath(line[1])
    image_right = get_image_from_sourcepath(line[2])
    
    images.extend([image_center, image_left, image_right])
    measurements.extend([steering_center, steering_left, steering_right])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch=4)

model.save('model.h5'.format(dataPath))