"""
This script is used to read images and cloud masks in bunches.

base_dir : directory with the regional data
cloud_dir : directory with the cloud masks
date_range : range of dates of interest

"""

from torch.utils.data import Dataset
import rasterio as rio
import numpy as np
from pathlib import Path
import os
       
class RegionalDataset(Dataset):
    def __init__(self,
                 base_dir,
                 cloud_dir,
                 date_range = None,
                 s2_channels = [3, 2, 1],
                 s2_range = [0, 1e4],
                 s2_scale = 2e3,
                 max_sar = 2,
                 db_sar = True,
                 filter_file = None,
                 shuffle = False
                ):
        
        self.base_dir = Path(base_dir)
        self.cloud_dir = Path(cloud_dir)
        self.all_dates = []
        for date in sorted(os.listdir(self.base_dir)):
            if date[:2] == '20':
                for roi in os.listdir(self.base_dir / date):
                    if roi[:3] == 'ROI':
                        self.all_dates.append(date + '/' + roi)
                        
        print('{} initial images found.'.format(len(self.all_dates)))
        
        self.date_range = date_range
        self.s2_channels = s2_channels
        self.s2_range = s2_range
        self.s2_scale = s2_scale
        
        self.max_sar = max_sar
        self.db_sar = db_sar
        
        if self.date_range is None:
            self.valid_dates = self.all_dates
        else:
            self.valid_dates = [d for d in self.all_dates if (d >= self.date_range[0] and d <= self.date_range[1])]  
            print('Filtering by dates: {} - {}'.format(*self.date_range))
            print('{} images remaining after date filtering'.format(len(self.valid_dates)))
            
        self.used_dates = []
        if filter_file is not None:
            with open(self.base_dir / filter_file) as f:
                for l in f:
                    if l.strip() in self.valid_dates:
                        self.used_dates.append(l.strip())
            print('{} final image files after filtering using file ""'.format(len(self.used_dates), filter_file))
        else:
            self.used_dates = self.valid_dates                            
                        
        self.cloud_files = os.listdir(self.cloud_dir)
        print('{} cloud masks found.'.format(len(self.cloud_files)))

        self.shuffle = shuffle
        self.shuffle_order = np.arange(len(self))
        np.random.shuffle(self.shuffle_order)
        
    def __len__(self):
        return len(self.used_dates)*len(self.cloud_files)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.shuffle_order[idx]        
        
        date_idx = idx // len(self.cloud_files)
        cloud_idx = idx % len(self.cloud_files)
        s2 = self.read_date(self.used_dates[date_idx])
        cm = np.load(self.cloud_dir / '{}.npy'.format(cloud_idx))
        return s2, cm
         
    def s2_preprocess(self, s2):
        s2clip = s2.clip(*self.s2_range)
        s2clip /= self.s2_scale
        return s2clip

    def read_date(self, DATE):
        s2root = self.base_dir / DATE / 'S2'
        s2_file = rio.open(s2root / os.listdir(s2root)[0])
        all_channels = s2_file.read()
        s2_stack = all_channels[self.s2_channels, ...].transpose(1, 2, 0)
        s2_stack = self.s2_preprocess(s2_stack)
        s2_file.close()
        return s2_stack