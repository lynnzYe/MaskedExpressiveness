from maskexp.util.tokenize_midi import extract_tokens_from_midi
from maskexp.definitions import DATA_DIR
import torch
from note_seq import PerformanceEvent as Perf
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.model.create_dataset import generate_maestro_dataset
import pickle
import tqdm
import os
import numpy as np
import json
import matplotlib.pyplot as plt


def stats_perf_distribution(dataset_pt, filter=None):
    if filter is None:
        filter = [Perf.NOTE_ON, Perf.VELOCITY, Perf.TIME_SHIFT]
    data_dict = torch.load(dataset_pt)

    if 'model_config_name' not in data_dict.keys():
        assert 'config' in data_dict.keys() and 'hotfix for compatibility with older dataset pt'
        data_dict['model_config_name'] = data_dict['config']

    perf_config = performance_model.default_configs[data_dict['config']]
    decoder = perf_config.encoder_decoder._one_hot_encoding
    train_ids, val_ids, test_ids = data_dict['train']['ids'], data_dict['val']['ids'], data_dict['test']['ids'],
    token_ids = torch.cat([train_ids, val_ids, test_ids])

    timeshift_mode = {
        'enabled': Perf.TIME_SHIFT in filter,
        'curr_dur': 0,
        'input_mode': False
    }

    perf_hist_dict = {str(k): [] for k in filter}
    for i, tlist in enumerate(tqdm.tqdm(token_ids)):
        prev_event = None
        for it, token in enumerate(tlist):
            event = decoder.decode_event(token)
            if timeshift_mode['enabled']:
                # FSM to parse duration
                if event.event_type != Perf.TIME_SHIFT and timeshift_mode['input_mode']:
                    perf_hist_dict[str(Perf.TIME_SHIFT)].append(timeshift_mode['curr_dur'])
                    timeshift_mode['curr_dur'] = 0
                    timeshift_mode['input_mode'] = False
            if timeshift_mode['enabled'] and event.event_type == Perf.TIME_SHIFT:
                # ==== Debug:
                if timeshift_mode['curr_dur'] > 0:
                    assert prev_event.event_type == Perf.TIME_SHIFT  # Continuation property
                    assert prev_event.event_value.item() == 100  # max utilization property
                # ==== Debug /
                timeshift_mode['curr_dur'] += event.event_value.item()
                timeshift_mode['input_mode'] = True
            else:
                for f in filter:
                    if event.event_type == f:
                        perf_hist_dict[str(f)].append(event.event_value.item())
            prev_event = event

        timeshift_mode = {
            'enabled': Perf.TIME_SHIFT in filter,
            'curr_dur': 0,
            'input_mode': False
        }
    return perf_hist_dict


def generate_huge_timevocab_data():
    generate_maestro_dataset(base_path='../../data/maestro-v3.0.0', save_name='mstro_big_time_vocab')


def analyze_stats():
    with open('data.pkl', 'rb') as pk:
        data = pickle.load(pk)
    # counts, bins = np.histogram(data[str(Perf.TIME_SHIFT)])
    # plt.stairs(counts, bins)
    ts_data =data[str(Perf.TIME_SHIFT)]
    plt.hist(ts_data, log=True)
    plt.title('Timeshift histogram')
    plt.savefig('hist.png')
    plt.show()

    # data_uniform = np.random.uniform(0, 100, 1000)
    # counts, bins = np.histogram(data_uniform)
    # plt.stairs(counts, bins)
    # plt.show()
    pass


def main():
    # stats = stats_perf_distribution(DATA_DIR + '/mstro_with_dyn.pt')
    # with open('data.pkl', 'wb') as pickle_file:
    #     pickle.dump(stats, pickle_file)

    analyze_stats()
    pass


if __name__ == '__main__':
    main()
