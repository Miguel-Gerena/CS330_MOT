# encoding: utf-8
"""
@author:  sherlock (changed by Nir)
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MOT17(ImageDataset):
    """MOT17.

    Reference:
        Milan, A., Leal-Taixé, L., Reid, I., Roth, S. & Schindler, K. MOT16: A Benchmark for Multi-Object Tracking. arXiv:1603.00831 [cs], 2016., (arXiv: 1603.00831)

    URL: `<https://motchallenge.net/data/MOT17/>`_

    Dataset statistics:
        - identities: ?
        - images: ?
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = ''  # 'https://motchallenge.net/data/MOT17.zip'
    dataset_name = "MOT17"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'MOT17-ReID')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"MOT17-ReID".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])

        super(MOT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):

        img_paths = glob.glob(osp.join(dir_path, '*.bmp'))

        # Refined pattern to match pid and camid
        pattern = re.compile(r'(\d+)_v_[a-zA-Z0-9_-]+_c(\d+)_\d+_acc_data\.bmp$')

        # data = []
        # for img_path in img_paths:
        #     print("img_path:", img_path)
        #     match = pattern.search(img_path)
        #     if match is not None:
        #         groups = match.groups()
        #         pid, camid = map(int, (groups[0], groups[2]))
        #         # Add conditions or adjustments as needed
        #         if pid == 0:  # Assuming 0 means background
        #             continue  # Ignore background images
        #         camid -= 1  # Adjusting camid index
        #         data.append((img_path, pid, camid))
        #     else:
        #         print(f"No match found for {img_path}")

        # print(data)
        pattern = re.compile(r'(\d+)_v_[a-zA-Z0-9_-]+_c(\d+)_\d+_acccon_data\.bmp$')

        data = []
        for img_path in img_paths:
            match = pattern.search(img_path)
            if match:
                pid, camid = map(int, match.groups())
                if pid == -1:
                    continue  # Junk images are ignored
                # assert 0 <= pid   # pid == 0 means background
                # assert 1 <= camid <= 5
                camid -= 1  # Index starts from 0
                if is_train:
                    pid = self.dataset_name + "_" + str(pid)
                    camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid))
                print("Match found!")
            else:
                # Handle cases where the pattern doesn't match the filename

                print(f"No match for {img_path}")
                print(pattern)

        return data
