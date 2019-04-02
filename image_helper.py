import numpy as np
import PIL.Image as img
import math
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

class ImageHelper:
    def open(self, name, size=None):
        image = img.open(name)

        if size is not None:
            self.image = image.resize(size, img.ANTIALIAS)
        else:
            self.image = image

        self.imageArray = np.array(self.image)
        if len(self.imageArray.shape) == 2:
            self.imageArray = np.reshape(self.imageArray, (self.imageArray.shape[0], self.imageArray.shape[1], 1))

    def openAsGray(self, name, size=None):
        image = img.open(name).convert('L')

        if size is not None:
            self.image = image.resize(size, img.ANTIALIAS)
        else:
            self.image = image

        self.imageArray = np.array(self.image)
        if len(self.imageArray.shape) == 2:
            self.imageArray = np.reshape(self.imageArray, (self.imageArray.shape[0], self.imageArray.shape[1], 1))

    def pad(self, multiple):
        # pad image so that its width and height are divisible by multiple
        self.extraWidth = (multiple - self.image.size[0] % multiple) % multiple
        self.extraHeight = (multiple - self.image.size[1] % multiple) % multiple
        print(self.imageArray.shape)
        self.imageArray = np.pad(self.imageArray, ((0, self.extraHeight), (0, self.extraWidth), (0, 0)), mode='constant')

    def getBlockArray(self, blocksize):
        self.pad(blocksize)
        return np.reshape(self.imageArray, (int(self.imageArray.size / (blocksize ** 2)), blocksize, blocksize))

    def getDCTMat(self, blocksize):
        dctMat = np.zeros((blocksize, blocksize))
        for i in range(blocksize):
            if i == 0:
                dctMat[i] = 1 / (blocksize ** 0.5)
            else:
                for j in range(blocksize):
                    dctMat[i, j] = ((2 / blocksize) ** 0.5) * math.cos(((2 * j + 1) * i * math.pi) / (2 * blocksize))
        return dctMat

    def dct(self, blocksize):
        self.pad(blocksize)
        arrReshaped = np.reshape(self.imageArray, (int(self.imageArray.size / (blocksize ** 2)), blocksize, blocksize))
        arrDCT = np.zeros(arrReshaped.shape)
        dctMat = self.getDCTMat(blocksize)
        arrReshaped = arrReshaped - 128

        for i in range(arrReshaped.shape[2]):
            arrDCT[i, :, :] = np.matmul(np.matmul(dctMat, arrReshaped[i, :, :]), dctMat.transpose())

        return arrDCT

    def invdct(self, blocksize, dct):
        arrReshaped = np.zeros(dct.shape)
        dctMat = self.getDCTMat(blocksize)

        for i in range(arrReshaped.shape[2]):
            arrReshaped[i, :, :] = np.matmul(np.matmul(dctMat.transpose(), dct[i, :, :]), dctMat)

        arrReshaped = arrReshaped - 128

        return arrReshaped

    def fromDCTForm(self, dctLike):
        assert dctLike.size == self.imageArray.size
        newHelper = ImageHelper()
        if self.imageArray.shape[2] != 1:
            newHelper.imageArray = np.reshape(dctLike, self.imageArray.shape)
        else:
            newHelper.imageArray = np.reshape(dctLike, self.imageArray.shape)[:, :, 0]
        newHelper.image = img.fromarray(newHelper.imageArray.astype('uint8'))

        return newHelper

    def saveImage(self, name):
        self.image.save(name)

