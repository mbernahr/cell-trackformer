# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT17 sequence dataset.
"""
import configparser
import csv
import os
from pathlib import Path
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List
import json
import pandas as pd

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import make_coco_transforms
from ..transforms import Compose


class DemoSequence(Dataset):
    """DemoSequence (MOT17) Dataset.
    """
    ####################################################################################
    # Modified DemoSequence to work as Container for the 'custom_dataset'

    DS_DIR = 'custom_dataset'

    def __init__(self, seq_name, root_dir: str = 'data', img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__()

        self._data_dir = os.path.join(root_dir, self.DS_DIR)

        train_dir = os.path.join(self._data_dir, 'train_sequences')
        test_dir = os.path.join(self._data_dir, 'test_sequences')

        self._train_folders = os.listdir(train_dir)
        self._test_folders = os.listdir(test_dir)
        self._seq_name = seq_name

        with open(os.path.join(self._data_dir, 'full_annotations.json'), 'r') as annotations_file:
            self.annotations_dict = json.load(annotations_file)

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))
        if (seq_name in self._train_folders):
            self.data = self._sequence(os.path.join(train_dir, seq_name))
        else:
            self.data = self._sequence(os.path.join(test_dir, seq_name))

        self.no_gt = False

    ####################################################################################

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f'{self._seq_name}'

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        ###########################################################################
        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])
        return sample
        ###########################################################################

    ###########################################################################
    # To handle our different structured dataset
    def _sequence(self, dir) -> List[dict]:
        total = []
        image_names = []
        for filename in sorted(os.listdir(dir)):
            extension = os.path.splitext(filename)[1]
            if extension in ['.png', '.jpg']:
                image_names.append(filename)
        images_df = pd.DataFrame(self.annotations_dict['images'])

        image_ids = images_df[images_df['file_name'].isin(image_names)]['id'].to_numpy()

        annotations_df = pd.DataFrame(self.annotations_dict['annotations'])
        annotations = {
            image_id:
                [annotation.to_dict() for _, annotation in
                 annotations_df[annotations_df['image_id'] == image_id].iterrows()]
            for image_id in image_ids
        }
        boxes = {}
        for id in image_ids:
            image_annotations = annotations[id]

            boxes[id] = {
                annotation['label']: np.array([int(value * 1024) for value in [
                    annotation['bbox'][0] - (annotation['bbox'][2] / 2),
                    annotation['bbox'][1] - (annotation['bbox'][3] / 2),
                    annotation['bbox'][0] + (annotation['bbox'][2] / 2),
                    annotation['bbox'][1] + (annotation['bbox'][3] / 2),
                ]])
                for annotation in image_annotations
            }

        total = [
            {
                'gt': boxes[id],
                'im_path': os.path.join(dir, images_df[images_df['id'] == id]['file_name'].iloc[0]),
                'vis': -1.0,
                'dets': [],
            }
            for id in image_ids
        ]

        return total

    ###########################################################################

    def load_results(self, results_dir: str) -> dict:
        return {}

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, Path(self._data_dir).name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])