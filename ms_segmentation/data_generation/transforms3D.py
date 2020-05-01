

import numpy as np
import torch
import random 
from scipy import ndimage


class RandomFlipX(object):
    """Flip 3D patch and labels in X direction.

    Parameters:
    p:  float
        probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if len(img)>1:
            image = img[0]
            labels = img[1]
            if random.random() < self.p:
                return np.flip(image, axis = -3).copy(), np.flip(labels, axis = -3).copy() #Flip around first 3D axis
            return img
        else:
            image = img[0]
            if random.random() < self.p:
                return np.flip(image, axis = -3).copy()
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomFlipY(object):
    """Flip 3D patch and labels in Y direction.

    Parameters:
    p:  float
        probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if len(img)>1:
            image = img[0]
            labels = img[1]
            if random.random() < self.p:
                return np.flip(image, axis = -2).copy(), np.flip(labels, axis = -2).copy() #Flip around first 3D axis
            return img
        else:
            image = img[0]
            if random.random() < self.p:
                return np.flip(image, axis = -2).copy()
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomFlipZ(object):
    """Flip 3D patch and labels in Z direction.

    Parameters:
    p:  float
        probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if len(img)>1:
            image = img[0]
            labels = img[1]
            if random.random() < self.p:
                return np.flip(image, axis = -1).copy(), np.flip(labels, axis = -1).copy() #Flip around first 3D axis
            return img
        else:
            image = img[0]
            if random.random() < self.p:
                return np.flip(image, axis = -1).copy()
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotationXY(object):


    def __init__(self, degrees, p=0.5):
        self.degrees = (-degrees, degrees)
        self.p = p

    @staticmethod
    def get_angle(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):

        if random.random() < self.p:
            if len(img)>1:
                image = img[0]
                labels = img[1]
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-3,-2), reshape=False), ndimage.rotate(labels, angle, axes = (-3,-2), reshape=False)
            else:
                image = img[0]
        
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-3,-2) )          
        return img
        

class RandomRotationYZ(object):


    def __init__(self, degrees, p=0.5):
        self.degrees = (-degrees, degrees)
        self.p = p

    @staticmethod
    def get_angle(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        if random.random() < self.p:
            if len(img)>1:
                image = img[0]
                labels = img[1]
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-2,-1), reshape=False), ndimage.rotate(labels, angle, axes = (-2,-1), reshape=False)
            else:
                image = img[0]
        
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-2,-1) )          
        return img

class RandomRotationXZ(object):


    def __init__(self, degrees, p=0.5):
        self.degrees = (-degrees, degrees)
        self.p = p

    @staticmethod
    def get_angle(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        if random.random() < self.p:
            if len(img)>1:
                image = img[0]
                labels = img[1]
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-3,-1), reshape=False), ndimage.rotate(labels, angle, axes = (-3,-1), reshape=False)
            else:
                image = img[0]
        
                angle = self.get_angle(self.degrees)
                return ndimage.rotate(image, angle, axes = (-3,-1) )          
        return img


class ToTensor3DPatch(object):

    def __call__(self, img):
        if len(img)>1:
            image = img[0]
            labels = img[1]

            return torch.Tensor(image), torch.Tensor(labels)
        else:
            return torch.Tensor(img[0])