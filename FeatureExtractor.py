__author__ = 'admin'
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from PIL import Image
from scipy import misc


class FeatureExtractor(object):
    """
    Extract feature for images.
    The output of the last fully connected layer of VGG net are used feature in this implementation

     Attributes:
        feature_model: VGG model
    """
    feature_model = Sequential()

    @classmethod
    def initialize(cls, model_path):
        """
        Initialize the VGG model
        :param model_path: File path of VGG model weight
        :return: None
        """
        cls.feature_model = cls.vgg_16(model_path)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        cls.feature_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    @classmethod
    def feature(cls, image):
        """
        Generate feature of a singe image
        :param image: Input image
        :return: A vector of feature
        """
        image = misc.imresize(image, (224,224,3))
        image = image.transpose((2,0,1))
        image = np.expand_dims(image, axis=0)
        return cls.feature_model.predict(image)

    @classmethod
    def iterate_feature(cls, proposals, image, axis=0):
        """
        Generate feature of images iterately
        :param proposals: Candidate object bounding boxes location and size
        :param image: Original image
        :param axis: X axis first or Y axis first
        :return: Feature matrix
        """
        feature = np.zeros((proposals.shape[0], 4096))
        for i in range(proposals.shape[0]):
            proposal = proposals[i]
            if axis == 0:
                img = image[proposal[0]:proposal[2], proposal[1]:proposal[3], :]
            else:
                img = image[proposal[1]:proposal[3], proposal[0]:proposal[2], :]
            img = misc.imresize(img, (224, 224, 3))
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            feature[i, :] = cls.feature_model.predict(img)[0]
        return feature

    @classmethod
    def batch_feature(cls, proposals, image, axis=0):
        """
        Generate feature for a batch of images
        However, this method will cost a large amount of memory.
        iterate_feature is recommended for limited amount of memory
        :param proposals: Candidate object bounding boxes location and size
        :param image: Original image
        :param axis: X axis first or Y axis first
        :return: Feature matrix
        """
        image_batch = np.zeros((proposals.shape[0], 3, 224, 224))
        for i in range(proposals.shape[0]):
            proposal = proposals[i]
            if axis == 0:
                img = image[proposal[0]:proposal[2], proposal[1]:proposal[3], :]
            else:
                img = image[proposal[1]:proposal[3], proposal[0]:proposal[2], :]
            img = misc.imresize(img, (224, 224, 3))
            img = img.transpose((2, 0, 1))
            image_batch[i, :, :, :] = img[:, :, :]
        return cls.feature_model.predict(image_batch)

    @classmethod
    def vgg_16(cls, weights_path=None):
        """
        Construct and compile the VGG net model
        :param weights_path: File path of VGG weight
        :return: VGG net model
        """
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        if weights_path:
            model.load_weights(weights_path)

        model.layers.pop()
        model.params.pop()
        model.layers.pop()
        return model


if __name__ == "__main__":
    im = Image.open('/Users/admin/Desktop/images00001.jpg')
    FeatureExtractor.initialize("/Users/admin/Documents/clothingTestImages/vgg16_weights.h5")
    f = FeatureExtractor.feature(im)
    f = FeatureExtractor.feature(im)
    print f.shape
