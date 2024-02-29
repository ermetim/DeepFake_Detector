import numpy as np
import os
import datetime
import time

# ML models
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from src.ml_functions import ImageProcessing
from src.ml_functions import convert_data
from src.ml_functions import save_model
from src.ml_functions import train_model

RANDOM_STATE = 12345
np.random.seed(RANDOM_STATE)

param_class = {
    'gray': True,
    'h': 144,
    'w': 144,
    'frac_h': 0.8,
    'frac_v': 0.5,
    'labels': {'real': 0, 'fake': 1},
}
param_foo = {
    'frames_path': os.path.join('datasets', 'frames'),
    'phases': ['train'],
    'sample_num': 1500,
}

train_sample = 1500
test_sample = 500
save_model_path = os.path.join('models1', 'ML')
hog_results = []
base_results = []

# train dataset
param_foo['sample_num'] = train_sample
param_foo['phases'] = ['train']
train_data = ImageProcessing(**param_class).get_images(**param_foo)
Xtrain = train_data['images']
Xtrain_hog = train_data['hog_images']
ytrain = train_data['targets']
print('*'*100)
print('Train dataset downloaded', len(Xtrain))


# test dataset
param_foo['sample_num'] = test_sample
param_foo['phases'] = ['test']
test_data = ImageProcessing(**param_class).get_images(**param_foo)
Xtest = test_data['images']
Xtest_hog = test_data['hog_images']
ytest = test_data['targets']
print('*'*100)
print('Test dataset downloaded', len(Xtest))

# Base pictures
Xtrain_pic, ytrain_pic = convert_data(Xtrain.copy(), ytrain.copy(), random_state=RANDOM_STATE)
Xtrain_hog, ytrain_hog = convert_data(Xtrain_hog.copy(), ytrain.copy(), random_state=RANDOM_STATE)
print('*'*100)
print('Train dataset converted')

Xtest_pic, ytest_pic = convert_data(Xtest.copy(), ytest.copy(), is_shuffle=False, random_state=RANDOM_STATE)
Xtest_hog, ytest_hog = convert_data(Xtest_hog.copy(), ytest.copy(), is_shuffle=False, random_state=RANDOM_STATE)
print('*'*100)
print('Test dataset converted')

# Train LogisticRegression
print('Training LogisticRegression on pictures in progress', '.'*20)
t0 = time.time()
acc_lr_pic, lr_model_pic, pred_lr_pic = train_model(Xtrain_pic, ytrain_pic,
                                                    Xtest_pic, ytest_pic,
                                                    LogisticRegression(solver='liblinear',
                                                                       random_state=RANDOM_STATE,
                                                                       class_weight='balanced')
                                                    )
# save best lr model for base pictures
save_model(lr_model_pic, file_name='lr_model_pic_best.pkl', path=save_model_path)
work_time_lr_pic = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_lr_pic', work_time_lr_pic)

# Train LGBM
print('Training LightGBM on pictures in progress', '.'*20)
t0 = time.time()
acc_lgbm_pic, lgbm_model_pic, pred_lgbm_pic = train_model(Xtrain_pic, ytrain_pic,
                                                          Xtest_pic, ytest_pic,
                                                          LGBMClassifier(random_state=RANDOM_STATE,
                                                                         class_weight='balanced',
                                                                         verbose=-1)
                                                          )
# save best lgbm model for base pictures
save_model(lgbm_model_pic, file_name='lgbm_model_pic_best.pkl', path=save_model_path)
work_time_lgbm_pic = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_lgbm_pic', work_time_lgbm_pic)

# Train RandomForestClassifier
print('Training RandomForestClassifier on pictures in progress', '.'*20)
t0 = time.time()
acc_rfc_pic, rfc_model_pic, pred_rfc_pic = train_model(Xtrain_pic, ytrain_pic,
                                                       Xtest_pic, ytest_pic,
                                                       RandomForestClassifier(random_state=RANDOM_STATE,
                                                                              class_weight='balanced')
                                                       )

# save best rfc model for base pictures
save_model(rfc_model_pic, file_name='rfc_model_pic_best.pkl', path=save_model_path)
work_time_rfc_pic = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_rfc_pic', work_time_rfc_pic)

# HOG pictures
# Train LogisticRegression
print('Training LogisticRegression on HOG in progress', '.'*20)
t0 = time.time()
acc_lr_hog, lr_model_hog, pred_lr_hog = train_model(Xtrain_hog, ytrain_hog,
                                                    Xtest_hog, ytest_hog,
                                                    LogisticRegression(solver='liblinear',
                                                                       random_state=RANDOM_STATE,
                                                                       class_weight='balanced')
                                                    )
# save best lr model for HOG pictures
save_model(lr_model_hog, file_name='lr_model_hog_best.pkl', path=save_model_path)
work_time_lr_hog = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_lr_hog', work_time_lr_hog)

# Train LGBM
print('Training LightGBM on HOG in progress', '.'*20)
t0 = time.time()
acc_lgbm_hog, lgbm_model_hog, pred_lgbm_hog = train_model(Xtrain_hog, ytrain_hog,
                                                          Xtest_hog, ytest_hog,
                                                          LGBMClassifier(random_state=RANDOM_STATE,
                                                                         class_weight='balanced',
                                                                         verbose=-1)
                                                          )
# save best lgbm model for HOG pictures
save_model(lgbm_model_hog, file_name='lgbm_model_hog_best.pkl', path=save_model_path)
work_time_lgbm_hog = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_lgbm_hog', work_time_lgbm_hog)

# Train RandomForestClassifier
print('Training RandomForestClassifier on HOG in progress', '.'*20)
t0 = time.time()
acc_rfc_hog, rfc_model_hog, pred_rfc_hog = train_model(Xtrain_hog, ytrain_hog,
                                                       Xtest_hog, ytest_hog,
                                                       RandomForestClassifier(random_state=RANDOM_STATE,
                                                                              class_weight='balanced')
                                                       )

# save best rfc model for HOG pictures
save_model(rfc_model_hog, file_name='rfc_model_hog_best.pkl', path=save_model_path)
work_time_rfc_hog = str(datetime.timedelta(seconds=int(time.time() - t0)))
print('work_time_rfc_hog', work_time_rfc_hog)


print('+'*100)
print(f'LR MODEL: accuracy for base pictures = {acc_lr_pic}, time={work_time_lr_pic}, accuracy for HOG pictures = {acc_lr_hog}, time={work_time_lr_hog}')
print(f'LGBM MODEL: accuracy for base pictures = {acc_lgbm_pic}, time={work_time_lgbm_pic}, accuracy for HOG pictures = {acc_lgbm_hog}, time={work_time_lgbm_hog}')
print(f'RFC MODEL: accuracy for base pictures = {acc_rfc_pic}, time={work_time_rfc_pic}, accuracy for HOG pictures = {acc_rfc_hog}, time={work_time_rfc_hog}')
print('+'*100)

labels = {'real': 0, 'fake': 1}
pred = np.round((pred_lgbm_pic + pred_rfc_pic + pred_lgbm_hog + pred_rfc_hog) / 4)
print('mean accuracy = ', accuracy_score(ytest_hog, pred))


if __name__ == "__main__":
    print("Done")
