import numpy as np
import os
import cv2
import pickle
from tqdm.auto import tqdm
from collections import defaultdict

# HOG
from skimage.feature import hog

# ML
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def save_model(model, file_name='default_model.pkl', path=os.path.join('models', 'ML')):
    """
    save model to .pkl file
    """
    try:
        os.makedirs(path)
    except:
        pass
    file_path = os.path.join(path, file_name)
    pickle.dump(model, open(file_path, 'wb'))


def load_model(file_name):
    """
    load model from .pkl file
    """
    file_path = os.path.join('models', file_name)
    return pickle.load(open(file_path, 'rb'))


def train_model(Xtrain, ytrain, Xtest, ytest, model):
    """
    Train model
    """
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtest)
    accuracy = accuracy_score(ytest, pred)
    return accuracy, model, pred


def convert_data(X,
                 y,
                 labels={'real': 0, 'fake': 1},
                 is_shuffle=True,
                 normalize=False,
                 random_state=12345
                 ):
    """
    Convert every picture to vector and encode labels
    :param X: numpy array of pictures
    :param y: numpy array of labels
    :param labels: dict of labels
    :param is_shuffle:  required shuffle or not
    :param normalize:  required normalize or not
    :return: numpy array of pictures and numpy array of labels
    """
    # Вытянем изображения в вектор
    X = np.asarray([el.ravel() for el in X])

    # закодируем
    y = [labels[item] for item in y]
    if is_shuffle:
        X, y = shuffle(X, y, random_state=random_state)
    if normalize is True:
        return np.array(X) / 255, np.array(y)
    return np.array(X), np.array(y)


class ImageProcessing():
    def __init__(self,
                 gray=True,
                 h=144,
                 w=144,
                 frac_h=0.8,
                 frac_v=0.5,
                 labels={'real': 0, 'fake': 1}
                 ):
        """
        :param gray:
        :param h:
        :param w:
        :param frac_h:
        :param frac_v:
        :param labels: {'real' : 0, 'fake' : 1} or ['real', 'fake']
        """
        self.frac_h = frac_h
        self.frac_v = frac_v
        self.gray = gray
        self.size = (h, w)
        self.labels = labels

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

    def get_images(self,
                   frames_path=os.path.join('datasets', 'frames'),
                   phases=['train'],
                   sample_num=None,
                   ):
        """
        sample_num = 10
        frames_path

        phases = ['train', 'val', 'test']
        """
        results = defaultdict(list)

        for phase in tqdm(phases, desc='phase'):

            for label in self.labels:
                folder_link = os.path.join(frames_path, phase, label)
                # Sample or all pictures
                if sample_num is None or sample_num >= len(os.listdir(folder_link)):
                    img_links = os.listdir(folder_link)
                else:
                    img_links = np.random.choice(os.listdir(folder_link), sample_num)

                filelist = [os.path.join(folder_link, i) for i in img_links]

                for fname in tqdm(filelist, desc='downloading pictures'):
                    img = cv2.imread(fname)
                    image, face_image, hog_image = self.transform_image(img)
                    results['images'].append(image)
                    results['face_images'].append(face_image)
                    results['targets'].append(label)
                    results['links'].append(fname)
                    results['hog_images'].append(hog_image)
        return results
