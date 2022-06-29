import os
import random
import torchvision
import numpy as np
from PIL import Image


def init_transform(targets, samples, keep_file='./txt/split_4000.txt', training=True):
    new_targets, new_samples = [], []
    if training and (keep_file is not None):
        assert os.path.exists(keep_file), 'keep file does not exist'
        with open(keep_file, 'r') as rfile:
            for line in rfile:
                indx = int(line.split('\n')[0])
                new_targets.append(targets[indx])
                new_samples.append(samples[indx])
    else:
        new_targets, new_samples = targets, samples
    return np.array(new_targets), np.array(new_samples)


class TransCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root,
        seed=123,
        keep_file=None,
        num_labeled=4000,
        training=True,
        transform=None,
        target_transform=None,
        supervised=True,
    ):
        super().__init__(root, training, None, None, True)
        self.supervised = supervised
        self.training = training
        self.target_transform = target_transform
        self.transform = transform

        if keep_file is not None:
            random.seed(seed)
            src = []
            num_classes = len(self.classes)
            shot = int(num_labeled / num_classes)
            for val in self.class_to_idx.values():
                cur_src = [index for index, value in enumerate(self.targets) if value == val]
                idxes = random.sample(cur_src, shot)
                for idx in idxes:
                    src.append(idx)

            with open(keep_file, "w") as f:
                for i in range(len(src)):
                    f.write(str(src[i]) + '\n')

        if self.supervised:
            self.targets, self.data = init_transform(self.targets, self.data, keep_file=keep_file, training=training)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            if self.supervised or not self.training:
                return self.transform(img), target
            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
                return img_1, img_2, target
        return img, target
