import csv
import cv2
import numpy as np

# dataPath: folder path where all IMG's and driving_log's are stored
dataPath = 'train'
driving_log_list = {
    'driving_log1.csv':'IMG1',
    'driving_log2.csv':'IMG2',
    'driving_log5.csv':'IMG5',
    'driving_log7.csv':'IMG7',
}

correction = 0.3 # this is a parameter to tune

def get_image_from_sourcepath(source_path, folder):
    filename = source_path.split('/')[-1]
    current_path = './{}/{}/{}'.format(dataPath,folder,filename)
    image = cv2.imread(current_path)
    return image

def read_lines_from_filename(filename):
    local_lines = []
    with open('./{}/{}'.format(dataPath, filename)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            local_lines.append(line)
    return local_lines

images = []
measurements = []

# lines: array that contains each row of the csv file
# line: row that contains the image path for images, and also the steering and throttle values associated.
# images: global array that contains all the images used to train the model as the input
# measurements: global array that contains all measurements used to train the model as the output
# correction: a parameter that needs to be tuned. It provides a correction in the scenario when the car sees the lane lines.
print('Reading from: ./{}/'.format(dataPath))
for (d_log, folder) in driving_log_list.items():
    print('Reading file: {}'.format(d_log))
    lines = read_lines_from_filename(d_log)

    for line in lines:
        steering_center = float(line[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        image_center = get_image_from_sourcepath(line[0], folder)
        image_left = get_image_from_sourcepath(line[1], folder)
        image_right = get_image_from_sourcepath(line[2], folder)
        
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
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch=3)

model.save('model.h5')