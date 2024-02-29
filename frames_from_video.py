from collections import defaultdict
import pandas as pd
import os
import cv2
import random
import uuid

from sklearn.model_selection import train_test_split
RANDOM_STATE = 12345
random.seed(RANDOM_STATE)

videos_folder = 'videos'
datasets_folder = 'datasets'
frames_folder = 'frames1'
phases = ['train', 'val', 'test']
classes = ['real', 'fake']


folders_path = {'datasets': os.path.join(os.getcwd(), datasets_folder),
                'videos': os.path.join(os.getcwd(), datasets_folder, videos_folder),
                'frames': os.path.join(os.getcwd(), datasets_folder, frames_folder)}


for phase in phases:
    folders_path[phase] = os.path.join(folders_path['frames'], phase)
    for class_name in classes:
        folders_path[phase+'_'+class_name] = os.path.join(folders_path[phase], class_name)


# check input folder
if not os.path.isdir(folders_path['videos']):
    name = os.path.split(folders_path['videos'])[-1]
    print(f'The folder "{name}" does not exist. Please check the path')

# Create folders
for folder in folders_path.keys():
    if not os.path.isdir(folders_path[folder]):
        os.mkdir(folders_path[folder])
        print(f'The folder "{os.path.split(folder)[-1]}" does not exist. The folder was created')

videos_path = defaultdict(list)

# Create list of video files links
folders = []
for dataset in os.listdir(folders_path['videos']):
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(folders_path['videos'], dataset)):
        folders.append(dirpath)

# all links of video files to dict by category 'fake' and 'real'
for folder in folders:
    if os.path.split(folder)[-1] in classes:
        for name in os.listdir(folder):
            class_name = os.path.split(folder)[-1]
            videos_path[class_name].append(os.path.join(folder, name))


def split_dataset(path_dict: dict) -> dict:
    """
    Split dataset to train(60%), val(20%), test(20%)
    :param path_dict: dict with keys 'train', 'val', 'test'
            and lists with links of files
    :return: dict with splited dataset paths
    """
    dataset_path_dict = defaultdict(dict)
    for name in classes:
        dataset_path_dict['train'][name], test = train_test_split(
            path_dict[name], test_size=0.4, random_state=RANDOM_STATE)
        dataset_path_dict['val'][name], dataset_path_dict['test'][name] = train_test_split(
            test, test_size=0.5, random_state=RANDOM_STATE)
    return dataset_path_dict


print('-'*100)
print("splitting dataset..............")

# split all files to train(60%), val(20%), test(20%)
dataset_path = split_dataset(videos_path)


def get_frames(path, new_path, nums=4):
    """
    Function to get frames from video file and save them to new folder
    :param path: the path of the folder where the video file is located
    :param new_path: new path of the folder for saving frames
    :param nums: number of frames to get from the video file
    """
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i, frame_number in enumerate(random.sample(range(1, frame_count), nums)):
        cap.set(1, frame_number)
        ret, frame = cap.read()
        frame_path = new_path + '-' + str(i) + '.jpg'
        cv2.imwrite(frame_path, frame)


print('-'*100)
print("Getting frames..............")

file_names = []
count = 0
for key1, values1 in dataset_path.items():
    for key2, values2 in values1.items():
        for path in values2:
            dataset_name = os.path.split(os.path.split(os.path.split(path)[0])[0])[-1]
            unique_filename = str(uuid.uuid4().hex)[:10]
            path_to = os.path.join(folders_path['frames'], key1, key2, dataset_name+'_'+key2+'_'+unique_filename)
            file_names.append(dataset_name+'_'+key2+'_'+unique_filename)
            get_frames(path, path_to, nums=4)
            count += 1
            if count % 500 == 0:
                print(count)

df = pd.DataFrame({'file_names': file_names})
df.to_csv('names.csv', index=False)

if __name__ == "__main__":
    print("Done")
