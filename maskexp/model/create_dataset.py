import os
from maskexp.util.prepare_data import prepare_token_data, mask_perf_tokens, save_datadict
from maskexp.magenta.models.performance_rnn import performance_model
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """

    # data_source: Sized
    # replacement: bool

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(47)

        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)

        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index - 1]

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}

    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])


class ResumableSequentialSampler(Sampler):
    r"""Samples elements sequentially.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.perm_index = 0
        self.perm = torch.arange(len(data_source))

    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        while self.perm_index < len(self.perm):
            yield self.perm[self.perm_index]
            self.perm_index += 1

    def __len__(self):
        return self.num_samples

    def get_state(self):
        return {"perm_index": self.perm_index}

    def set_state(self, state):
        self.perm_index = state["perm_index"]


def load_dataset(pt_path, batch_size=32):
    """
    Read a pt file containing the train tokens
    :param pt_path: { train: {ids, masks, labels}, val: {...}, test: {...} }
    :return:
    """
    data = torch.load(pt_path)

    train_ids, train_msks, train_labels = data['train']['ids'], data['train']['msks'], data['train']['labels']
    val_ids, val_msks, val_labels = data['val']['ids'], data['val']['msks'], data['val']['labels']
    test_ids, test_msks, test_labels = data['test']['ids'], data['test']['msks'], data['test']['labels']

    train_dataset = TensorDataset(train_ids, train_msks, train_labels)
    val_dataset = TensorDataset(val_ids, val_msks, val_labels)
    test_dataset = TensorDataset(test_ids, test_msks, test_labels)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def generate_maestro_dataset(base_path, basic_config=None):
    if basic_config is None:
        basic_config = {
            'min_seq_len': 64,
            'max_seq_len': 128,
            'model_config_name': 'performance_with_dynamics'
        }
    paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                paths.append(os.path.join(root, file))

    perf_config = performance_model.default_configs[basic_config['model_config_name']]

    data_dict = prepare_token_data(paths, perf_config=perf_config,
                                   min_seq_len=basic_config['min_seq_len'],
                                   max_seq_len=basic_config['max_seq_len'])
    data_dict['config'] = 'performance_with_dynamics'
    data_dict = data_dict | basic_config
    save_datadict(data_dict, save_name='mstro_with_dyn')


if __name__ == '__main__':
    # generate_maestro_dataset('../../data/maestro-v3.0.0')
    pass
