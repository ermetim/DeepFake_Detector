import torch
import os
import pandas as pd
import numpy as np
import time
import datetime
import cv2
import copy
from tqdm import tqdm
from torchvision import transforms


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, dataloaders, model_path, epochs=2, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.epochs = epochs
        self.device = torch.device(device)
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0.0
        self.losses = {'train': [], 'val': []}
        self.accuracy = {'train': [], 'val': []}
        self.datasets_sizes = {phase: len(self.dataloaders[phase].dataset) for phase in self.dataloaders}
        self.model = self.model.to(self.device)
        self.model_path = model_path

    def train(self):
        start = time.time()
        for epoch in tqdm(range(self.epochs), desc='Epoch'):
            t0_epoch = time.time()
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(self.dataloaders[phase], leave=False, desc=f'{phase} iter'):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.datasets_sizes[phase]
                epoch_accuracy = running_corrects.double() / self.datasets_sizes[phase]

                self.losses[phase].append(epoch_loss)
                self.accuracy[phase].append(epoch_accuracy.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')

                if phase == 'val' and epoch_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_accuracy
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

                    # save current best model
                    model_file_name = os.path.join(self.model_path, f'model_epoch_{epoch}.pth')
                    torch.save(self.model, model_file_name)
                if phase == 'train':
                    self.scheduler.step()

            epoch_time = str(datetime.timedelta(seconds=int(time.time() - t0_epoch)))
            print(f'epoch # {epoch}: elapsed time {epoch_time}')

            # save loss and score
            pd.DataFrame(self.losses).to_csv(os.path.join(self.model_path, 'loss.csv'), index=False)
            pd.DataFrame(self.accuracy).to_csv(os.path.join(self.model_path, 'score.csv'), index=False)

        time_elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))

        print(f'Training complete in {time_elapsed}')
        print('Best val Acc: {:.4f}'.format(self.best_accuracy))

        # save best model
        model_file_name = os.path.join(self.model_path, f'best_model.pth')
        self.model.load_state_dict(self.best_model_wts)
        # torch.save(self.model.state_dict(), model_file_name)
        torch.save(self.model, model_file_name)

        return self.model, self.losses, self.accuracy

    def get_best_model_wts(self):
        return self.best_model_wts

    def evaluate(self):
        self.model.eval()

        curr_correct = 0

#         for inputs, labels in self.dataloaders['test']:
        for inputs, labels in self.dataloaders['test']:
            inputs, labels = tqdm((inputs.to(self.device), labels.to(self.device)), desc='batches')

            output = self.model(inputs)
            _, preds = torch.max(output, 1)

            curr_correct += torch.sum(preds == labels)

        return curr_correct.double() / self.datasets_sizes['test']

    def predict(self, dataloader_test, class_names=None, is_probs=True):
        probs = []
        self.model.eval()

        with torch.no_grad():
            for inputs in tqdm(dataloader_test):
                inputs = inputs.to(self.device)
                preds = self.model(inputs)
                preds = preds.cpu()
                probs.append(preds)

        probs = torch.cat(probs)
        probs = torch.nn.functional.softmax(probs, dim=-1).numpy()

        if is_probs is False:
            probs = np.argmax(probs, axis=1)
        if class_names is None:
            return probs
        probs = class_names[probs]

        return probs


def transform_image(path, transformer=None):

    """
    Prepare an image for the model

    :param path: image path
    :param transformer: torchvision.transforms
    :return: torch.Size([1, 1, 3, 224, 224])
    """
    # download the picture by OpenCV
    image = cv2.imread(path)

    # transform image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transformer
    if transformer is None:
        transformer = transforms.Compose([transforms.ToTensor()])

    # transformation the image
    transformed_image = transformer(image_rgb)

    # Add 2 additional dimentions (batch size, batch numbers)
    transformed_image = transformed_image.unsqueeze(0).unsqueeze(0)
    return transformed_image
