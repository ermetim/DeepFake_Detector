import cv2
import numpy as np
from skimage.feature import hog

class ImageProcessing():
    def __init__(self,
                 gray=True,
                 H = 144,
                 W = 144,
                 frac_h = 0.8,
                 frac_v = 0.5
                 ):
        self.frac_h = frac_h
        self.frac_v = frac_v
        self.gray = gray
        self.size = (H, W)

    def get_face(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        face_img = face_cascade.detectMultiScale(image)
        if len(face_img) != 1:
            face = image
        else:
            for (a, b, w, h) in face_img:
                c_a = min(a, w)
                c_a = int(c_a - c_a * self.frac_h)
                c_b = min(b, h)
                c_b = int(c_b - c_b * self.frac_v)
                face = image[b - c_b: b + h + c_b, a - c_a: a + w + c_a]
        face = cv2.resize(face, self.size, interpolation=cv2.INTER_AREA)
        return np.array(face)

    def get_hog(self, image):
        if len(image.shape) == 2:
            channel_axis = None
        else:
            channel_axis = 2
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=channel_axis)
        #         hog_image = cv2.hog_image(hog_image, self.size, interpolation = cv2.INTER_AREA)
        return np.array(hog_image)

    def transform_image(self, image):
        if self.gray is True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_image = self.get_face(image)

        hog_image = self.get_hog(face_image)

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        return image, face_image, hog_image
