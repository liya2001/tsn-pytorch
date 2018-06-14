import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

from config import get_list_file


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def offset(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, mode,
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None, random_shift=True):

        if mode not in ('train', 'val', 'test'):
            raise ValueError('Mode must be in train, val or test!')
        if modality not in ('RGB', 'Flow'):
            raise ValueError('Modality must be RGB or Flow!')

        self.root_path = root_path
        self.mode = mode
        self.list_file = get_list_file(mode, modality)
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift

        self._parse_list()

    def _load_image(self, record, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(self.root_path, record.path, '{}.jpg'.format(idx+record.offset))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, record.path, '{}-x.jpg'.format(idx+record.offset))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, record.path, '{}-y.jpg'.format(idx+record.offset))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        Sample load indices
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        # It seems TSN didn't sue validate data set,
        # our val data set is equivalent to TSN model's test set
        if self.mode == 'train':
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_path = '/home/liya/workspace/trecvid/data/candidate_region'
    train_flow = TSNDataSet(root_path=data_path, mode='train', modality='Flow')
    trian_loader = DataLoader(train_flow, batch_size=4)

