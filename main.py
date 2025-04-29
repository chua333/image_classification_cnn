import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# normalizing the images to a range of 0 to 1
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# for more reference visit -> https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# # to show the images
# for i in range(10):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])
# plt.show()

# # this will reduce the size of the dataset
# training_images = training_images[:30000]
# training_labels = training_labels[:30000]
# testing_images = testing_images[:6000]
# testing_labels = testing_labels[:6000]

# ##################################################################################################################
# # this part is to train a model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels, verbose=2)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# model.save('cifar10_model.keras')
# # model.save_weights('cifar10_weights.h5')

# ##################################################################################################################

# this part is to load a model
model = models.load_model('cifar10_model.keras')

# create a function that gets all image files and resize them to 32x32
def get_images_from_folder(folder='.'):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            img = cv.imread(filepath)
            if img is not None:
                img = cv.resize(img, (32, 32))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                images.append((img, filename))
    return images

images_to_predict = get_images_from_folder()
for img, name in images_to_predict:
    prediction = model.predict(np.array([img / 255.0]), verbose=0)
    index = np.argmax(prediction)
    print(f"Prediction for {name}: {class_names[index]}")
