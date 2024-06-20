# TODO: reference original repo

'''
Script that converts the labelbox downloaded dataset
and creates a training and testing set. It loades the images,
patches them, and saves into one data structure prepared for
the use by dataset.py

'''
import h5py
import numpy as np
import logging
from tqdm import tqdm
import os
import random
import csv
from collections import Counter
import argparse

from sklearn.model_selection import train_test_split

import torch
from torch.nn import Unfold, ReflectionPad2d
'''
Parameters:
- label: which label(s) to include in the instances (patch, mask(s))
- patch_size: size of a single patch extracted from the image
- stride: determines sampling frequency (we want img_size % stride == 0)
- added_pad: additional padding to accomodate rotation later
- resize: change magnification of the image (used for some classes)
- test_set_size: percentage if images used for the test set

Input:
h5 dataset with images and masks

Output:
train dataset:
h5 file with N (channels x ps_train x ps_train) arrays with corresponding masks

test set:
h5 file with M (labels x ps_test x ps_test) arrays with corresponding masks

meta_info:
ids of images in the train/test sets
etc.
'''
seed = 42
img_size = (1024, 1024, 4)


labels = {
    'Tubuli': 1,
    'Vein': 2,
    'Vessel_indeterminate': 2,  # Same as Vein - relabelled
    'Artery': 3,
    'Glomerui': 4
}


class CreateDataset:
    def __init__(self, h5_path, out_path, labels=labels,
                 patch_size=512, stride=512, added_pad=0):

        assert os.path.exists(out_path)

        self.h5_path = h5_path
        self.out_path = out_path
        self.labels = labels
        self.patch_size = patch_size
        self.stride = stride
        self.added_pad = added_pad
        self.ppi = (1024 // stride) ** 2    # Patches per image

        # Create Unfold and ReflectionPad instances for train/test patching
        self.unfold_img = Unfold(kernel_size=patch_size + (2 * added_pad), stride=stride)
        self.unfold_lbl = Unfold(kernel_size=patch_size, stride=stride)
        self.pad = ReflectionPad2d(added_pad)

    def _data_split(self):

        ids = list()
        with h5py.File(self.h5_path, 'r') as h5:
            for name, cut in h5.items():
                if any([x in cut.keys() for x in labels.keys()]):
                    if not name.endswith('_0'):     # Omit repeated annotations
                        ids.append(name)
                else:
                    logging.info(f'No label data for:\n{name}')
        logging.info("Images IDs:\n{}".format('\n'.join(ids)))

        # unique_slides = Counter([x.split(' ')[0] for x in ids])

        # test_slides = [x[0] for x in unique_slides.most_common(5)]
        ids_train, ids_test = [], []
    
        # randomly split the data with a seed
        random.seed(seed)
        for x in ids:
            if random.random() < 0.075:
                ids_test.append(x)
            else:
                ids_train.append(x)
        
        # for x in ids:
        #     if x.split(' ')[0] in test_slides:
        #         ids_test.append(x)
        #     else:
        #         ids_train.append(x)

        # Using the slide with the most patches as the test set
        # test_slide = unique_slides.most_common(1)[0][0]

        # ids_train, ids_test = [], []
        # for x in ids:
        #     if test_slide in x:
        #         ids_test.append(x)
        #     else:
        #         ids_train.append(x)

        self.ids_train = ids_train
        self.ids_test = ids_test

    # def _get_outcome_dict(self, csv_path):
    #     outcome_dict = {}
    #     with open(csv_path, 'r') as file:
    #         reader = csv.reader(file)
    #         for row in reader:
    #             outcome_dict[row[1]] = 1 if row[2] in pos_outcomes else 0
    #     self.outcome_dict = outcome_dict

    def _create_datasets_files(self, train=True):
        '''
        Fixes the memory address of the data
        '''
        if train:
            tps = self.patch_size + (2 * self.added_pad)
            N = len(self.ids_train) * self.ppi
            ds = (N, tps, tps, 3)
            ls = (N, self.patch_size, self.patch_size)
            path = self.out_path + '/train_data_1024.h5'
        else:
            N = len(self.ids_test)
            ds = (N, 1024, 1024, 3)
            ls = (N, 1024, 1024)
            path = self.out_path + '/test_data_1024.h5'

        # Creating empty dataset
        with h5py.File(path, 'w') as h5:
            data = h5.create_dataset('data', shape=ds, dtype=np.uint8)
            labels = h5.create_dataset('labels', shape=ls, dtype=np.uint8)

    def _populate_dataset(self, train=True):
        if train:
            ids = self.ids_train
            path = self.out_path + '/train_data_1024.h5'
        else:
            ids = self.ids_test
            path = self.out_path + '/test_data_1024.h5'

        file = h5py.File(path, 'r+')

        with h5py.File(self.h5_path, 'r') as h5:

            with tqdm(total=len(ids), desc='Images Processed', unit='img') as pbar:
                # For each training image in the file
                i = 0
                for index in ids:

                    # Get raw image and combine masks
                    img = np.array(h5[index].get('rawimage'))

                    if img.shape[2] == 4:   # Get RGB data if images have alpha channel
                        img = img[:, :, :3]

                    labelmap = np.zeros((1024, 1024))
                    for labelname in self.labels.keys():
                        if labelname in h5[index].keys():
                            mask = np.array(h5[index].get(labelname))
                            labelmap[mask > 0] = self.labels[labelname]

                    if labelmap.sum() == 0:    # Check if label is not empty
                        print('Empty label:', index)

                    if train:
                        # Pad and Patch the image and mask(s)
                        patch_img, patch_label = self.pad_and_patch(img, labelmap)

                        # Populate the dataset
                        file['data'][i:i + self.ppi] = patch_img
                        file['labels'][i:i + self.ppi] = patch_label
                        i += self.ppi

                    else:
                        file['data'][i] = img
                        file['labels'][i] = labelmap
                        i += 1

                    pbar.update(1)

        file.close()

    def pad_and_patch(self, img, masks):
        '''
        - in:
        img: ndarray (ic, iw, ih)
        masks: ndarray (nl, iw, ih)
        - out:
        tuple: (patches, labels)
        '''

        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)   # Dims for patching
        masks = torch.from_numpy(masks).float().unsqueeze(0).unsqueeze(0)   # Dims for patching

        ps = self.patch_size + (2 * self.added_pad)
        img = self.pad(img)

        if self.patch_size == 1024:
            patched_img = img.permute(0, 2, 3, 1)\
                .numpy()
            patched_masks = masks.squeeze(0)\
                .numpy()
        else:
            patched_img = self.unfold_img(img).squeeze(0)\
                .permute(1, 0)\
                .view(-1, 3, ps, ps)\
                .permute(0, 2, 3, 1)\
                .numpy()

            patched_masks = self.unfold_lbl(masks).squeeze(0)\
                .permute(1, 0)\
                .view(-1, self.patch_size, self.patch_size)\
                .numpy()

        return (patched_img, patched_masks)

    def run(self):

        logging.info('Making the data split')
        self._data_split()

        # logging.info('Creating the outcome dictionary')
        # self._get_outcome_dict(csv_path)

        logging.info(f'List of test IDs:\n' + '\n'.join(self.ids_test))

        logging.info('Creating datasets')
        self._create_datasets_files(train=True)
        self._create_datasets_files(train=False)

        logging.info('Populating new datasets')
        self._populate_dataset(train=True)
        self._populate_dataset(train=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='data/results.h5',
                        dest='input', help="Specify the h5 input file")
    parser.add_argument('--output', '-o', dest='output', default='data/',
                        help='Output directory')
    parser.add_argument('--patch-size', '-ps', dest='ps', type=int, default=512,
                        help='Train patch size')
    parser.add_argument('--stride', '-s', dest='s', type=int, default=512,
                        help='Train patch size')
    parser.add_argument('--padding', '-p', dest='p', type=int, default=0,
                        help='Added mirror padding amount')

    return parser.parse_args()


if __name__ == '__main__':
    # args = get_args()

    # logging.basicConfig(filename=args.output + '/data_prep.log',
    #                     filemode='a',
    #                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                     datefmt='%H:%M:%S',
    #                     level=logging.DEBUG)

    # creator = CreateDataset(h5_path=args.input,
    #                         out_path=args.output,
    #                         patch_size=args.ps,
    #                         stride=args.s,
    #                         added_pad=args.p)

    print('Running the dataset creation')

    logging.basicConfig(filename='/vol/bitbucket/dks20/renal_ssn/labelbox_download/data_prep.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    creator = CreateDataset(h5_path='/vol/bitbucket/dks20/renal_ssn/labelbox_download/results.h5',
                            out_path='/vol/bitbucket/dks20/renal_ssn/labelbox_download/',
                            patch_size=1024,
                            stride=1024,
                            added_pad=0)
    creator.run()