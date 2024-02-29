# Imports
import torch
import torchvision
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import os
import numpy as np

from src.dl_functions import ModelTrainer

# For local
DEFAULT_PATH = ''
H = 144
W = 144

# Set default size for pictures
DEFAULT_SIZE = (H, W)  # for transform

# Create full path for target directory
DATASET_PATH = os.path.join(DEFAULT_PATH, 'datasets', 'frames')
MODELS_PATH = os.path.join(DEFAULT_PATH, 'models', 'DL')

TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH = os.path.join(DATASET_PATH, 'val')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

# Set random state number
RANDOM_STATE = 12345

# Defaults for training
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 5e-4
EMBEDDING_DIM = 4
EPSILON = 1e-6

MEAN = np.array([0.42256699, 0.40667898, 0.40163194])
STD = np.array([0.26578207, 0.26359828, 0.27205688])

# Select device "GPU" or "CPU"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the folder "models"
try:
    os.makedirs(MODELS_PATH)
    print(f'The folder "{MODELS_PATH}" was created.')
except:
    print(f'The folder "{MODELS_PATH}" is available.')

# Transformation pictures with augmentation
TRANSFORM_TRAIN = transforms.Compose([transforms.RandomApply([transforms.RandomHorizontalFlip(p=0.5),
                                                              transforms.RandomVerticalFlip(p=0.5),
                                                              transforms.ColorJitter(hue=(0.1, 0.2)),
                                                              transforms.RandomRotation(degrees=30, fill=0)], p=0.5),
                                      transforms.Resize(DEFAULT_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])

TRANSFORM_VAL = transforms.Compose([transforms.Resize(DEFAULT_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN, STD)])

TRANSFORM_TEST = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(DEFAULT_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)])

# Loading train and validation data with target transformation
# Initialize the dataset and data loaders
train_data = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                              transform=TRANSFORM_TRAIN)

val_data = torchvision.datasets.ImageFolder(root=VAL_PATH,
                                            transform=TRANSFORM_VAL)

test_data = torchvision.datasets.ImageFolder(root=TEST_PATH,
                                             transform=TRANSFORM_VAL)

# Create train and val loaders
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           drop_last=True)

val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          drop_last=False)

# union all dataloaders in one dict
dataloaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}

# labeling of classes
class_names = train_data.class_to_idx

# Quantity of classes for target transformation
num_classes = len(class_names)

# Make dictinary of classes for decoding from prediction
class_names_dict = dict(zip(class_names.values(), class_names.keys()))

# Download default model
model_efficient = models.efficientnet_v2_m(weights="DEFAULT")


# Get the input features of the classifier
num_ftrs = model_efficient.classifier[1].in_features
# num_ftrs = 1280

for param in model_efficient.parameters():
    param.require_grad = False

# Change number of output features in model efficientnet
model_efficient.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                           nn.Linear(num_ftrs, len(class_names)))  # len(class_names) Number of classes

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_efficient.parameters(), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

trained_model = ModelTrainer(model=model_efficient,
                             criterion=loss_func,
                             optimizer=optimizer,
                             scheduler=exp_lr_scheduler,
                             epochs=NUM_EPOCHS,
                             dataloaders=dataloaders,
                             device=DEVICE,
                             model_path=MODELS_PATH)

model_efficient, losses_efficient, accuracy_efficient = trained_model.train()


if __name__ == "__main__":
    print("Done")
