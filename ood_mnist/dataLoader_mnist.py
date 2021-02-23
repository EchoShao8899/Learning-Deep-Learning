from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import random


class ClassFilterSampler(Sampler):
    def __init__(self, data_source, label_range, label_pos, shuffle):
        self.data_source = data_source
        self.label_range = label_range
        self.label_pos = label_pos
        self.shuffle = shuffle
        self.len = 0

    def __iter__(self):
        total_len = len(self.data_source)
        l = list(range(total_len))
        if self.shuffle == True:
            random.shuffle(l)
        data = []
        for index in l:
            if self.data_source.__getitem__(index)[self.label_pos] in self.label_range:
                self.len += 1
                data.append(index)
        return iter(data)

    def __len__(self):
        return self.len


def ClassFilterDataloader(data_source, label_range, label_pos, shuffle, batch_size):
    """
        Choose data with specific labels and provide iterator over this subset of data
        Arguments:
            data_source (Dataset): dataset from which to load the initial data
            label_range (list): a list containing your expected label
            label_pos (int): show which entry is the data label in data_source.__getitem__()
            shuffle (bool):  whether you want to shuffle your data
            batch_size (int): how many samples per batch to load
    """
    sampler = ClassFilterSampler(data_source,label_range,label_pos, shuffle)
    return DataLoader(data_source, batch_size=batch_size, sampler=sampler)
