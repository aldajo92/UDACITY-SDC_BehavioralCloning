import csv
import cv2
import numpy as np

# dataPath: folder path where all IMG's and driving_log's are stored
dataPath = 'train/'

lines = []
print('Reading from: ./{}'.format(dataPath))
with open('./{}driving_log.csv'.format(dataPath)) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# images: global list that contains all the images used to train the model as the input
# measurements: global list that contains all measurements used to train the model as the output
images = []
measurements = []

# lines: list that contains each row of the csv file
# line: row that contains the image path for images, and also the steering and throttle values associated, as a list.
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './{}IMG/{}'.format(dataPath,filename)
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, Y_train, validation_split = 0.2, shuffle = True, nb_epoch=2)

model.save('model.h5')