import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, preprocess,
                 experiment_indx=0, few_shot=None):

        self.load_function = load_function
        self.preprocess = preprocess
        self.phase = phase
        self.few_shot = few_shot

        assert k_shot in [0, 1, 5, 10]
        assert experiment_indx in [0, 1, 2]

        self.category = category

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot,
                                                                                   experiment_indx)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot, experiment_indx):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                      k_shot,
                                                                                                      experiment_indx)

        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        cv2_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = self.preprocess(Image.fromarray(cv2_img))
        
        # pil_img = Image.open(img_path).convert('RGB')
        # img = self.preprocess(pil_img)

        if gt == 0:
            gt = np.zeros([img.shape[-2], img.shape[-1]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (img.shape[-2], img.shape[-1]), interpolation=cv2.INTER_NEAREST)
            gt[gt > 0] = 255

        # img = cv2.resize(img, (1024, 1024))

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'
        
        if not self.few_shot:
            return img, gt, label, img_name, img_type
        else:
            # load 1-shot normal_image as ref_img
            train_img_folder = os.path.join(img_path.rsplit('/', 3)[0], 'train', 'good')
            train_imgs = os.listdir(train_img_folder)
            train_img_name = random.sample(train_imgs, 1)[0]
            train_img_path = os.path.join(train_img_folder, train_img_name)
            
            train_cv2_img = cv2.imread(train_img_path, cv2.IMREAD_COLOR)
            train_img = self.preprocess(Image.fromarray(train_cv2_img))
            
            return img, gt, label, img_name, img_type, train_img
