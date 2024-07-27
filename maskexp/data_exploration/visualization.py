from maskexp.util.tokenize_midi import extract_tokens_from_midi
from maskexp.definitions import DATA_DIR
import torch


def stats_time_shift(dataset_pt):
    data = torch.load(dataset_pt)
    train_ids, val_ids, test_ids = data['train']['ids'], data['val']['ids'], data['test']['ids'],
    token_ids = torch.cat([train_ids, val_ids, test_ids])

    pass


if __name__ == '__main__':
    stats_time_shift(DATA_DIR + '/data/mstro_with_dyn.pt')
    pass
