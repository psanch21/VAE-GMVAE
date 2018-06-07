import numpy as np


class Dataset:
    def __init__(self, data, labels, flat):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = data.shape[0]
        self.height = data.shape[1]
        self.width = data.shape[2]

        self._num_channels = data.shape[-1]
        self.set_flat(flat)
        pass

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels
    def set_flat(self, flat):
        if(not flat):
            self._data = np.reshape(self._data, [-1, self.height, self.width, self._num_channels])
        else:
            self._data = np.reshape(
                self._data, [-1, self.height * self.width * self._num_channels])

    def get_shape(self, flat=False):
        if flat:
            return (None, self.height * self.width * self._num_channels,)
        else:
            return (None, self.height, self.width, self._num_channels)

    def get_dim(self, flat=False):
        return self.height * self.width * self._num_channels

    def get_epochs_completed(self):
        return self._epochs_completed

    def set_epochs_completed(self, epochs):
        self._epochs_completed = epochs

    def num_batches(self, batch_size):
        return int(self._num_examples / batch_size)

    def reset(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        return

    def next_batch(self, batch_size, shuffle=True):
        '''
        Return the next batch and keeps track of number of epochs completed.
        '''
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self._data[idx]  # get list of `num` random samples
            self._labels = self._labels[idx]
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples
            self._labels = self._labels[idx0]

            start = 0
            # avoid the case where the #sample != integar times of batch_size
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]
