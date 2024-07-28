from maskexp.model.create_dataset import load_dataset
from maskexp.definitions import DATA_DIR
from maskexp.util.prepare_data import mask_perf_tokens
from tools import ExpConfig, load_ckpt, print_perf_seq, decode_perf_logits, hits_at_k, \
    accuracy_within_n, bind_metric
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.model.bert import NanoBertMLM
from pathlib import Path
import torch
import functools
import math
import matplotlib.pyplot as plt
import note_seq
import os
import tqdm
import time

NDEBUG = True


def test_mlm(model, perf_config, test_dataloader, metrics=None, device=torch.device('mps')):
    if metrics is None:
        metrics = [hits_at_k, accuracy_within_n]

    test_metrics = {metric.__name__: [] for metric in metrics}
    tokenizer = perf_config.encoder_decoder
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(test_dataloader)):
            input_ids = batch[0]
            attention_mask = batch[1]

            # Mask tokens
            inputs, labels = mask_perf_tokens(input_ids, perf_config=perf_config, mask_prob=0.15,
                                              special_ids=(note_seq.PerformanceEvent.VELOCITY,))
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()

            # Calculate and store each metric
            for metric in metrics:
                metric_value = metric(logits, labels)
                if not math.isnan(metric_value):
                    test_metrics[metric.__name__].append(metric_value)

            if step == len(test_dataloader) - 1:
                decoded = [tokenizer.class_index_to_event(i, None) for i in inputs[0, :].tolist()]
                print_perf_seq(decoded)
                print_perf_seq(decode_perf_logits(logits, tokenizer._one_hot_encoding))

            if not NDEBUG:
                break

    avg_test_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {avg_test_loss}')

    # Average out metrics over all batches
    avg_test_metrics = {metric: sum(values) / len(values) for metric, values in test_metrics.items()}
    for metric_name, avg_value in avg_test_metrics.items():
        print(f'{metric_name}: {avg_value}')

    # Save the checkpoint
    return avg_test_loss, avg_test_metrics


def run_mlm_test(test_settings: ExpConfig = None):
    if test_settings is None:
        test_settings = ExpConfig()
    perf_config = performance_model.default_configs[test_settings.perf_config_name]

    _, _, test = load_dataset(test_settings.data_path)

    model = NanoBertMLM(vocab_size=perf_config.encoder_decoder.num_classes,
                        n_embed=test_settings.n_embed,
                        max_seq_len=test_settings.max_seq_len,
                        n_layers=test_settings.n_layers,
                        n_heads=test_settings.n_heads,
                        dropout=test_settings.dropout)
    model.to(test_settings.device)
    load_ckpt(model, None, f'{test_settings.save_dir}/checkpoints/{test_settings.model_name}.pth')

    hits_1, hits_3, hits_5 = (bind_metric(hits_at_k, k=1),
                              bind_metric(hits_at_k, k=3),
                              bind_metric(hits_at_k, k=5))
    acc_wi_1, acc_wi_2, acc_wi_3 = (bind_metric(accuracy_within_n, n=1),
                                    bind_metric(accuracy_within_n, n=2),
                                    bind_metric(accuracy_within_n, n=3))

    avg_test_loss, metrics = test_mlm(model, perf_config, test,
                                      metrics=[hits_1, hits_3, hits_5, acc_wi_1, acc_wi_2, acc_wi_3]
                                      )
    metrics['loss'] = avg_test_loss

    torch.save(metrics,
               f'/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/logs/{test_settings.model_name}-metrics.pth')
    return


def test_velocitymlm():
    st = ExpConfig(model_name='velocitymlm',
                   save_dir='/Users/kurono/Documents/python/GEC/ExpressiveMLM/save',
                   data_path='/Users/kurono/Documents/python/GEC/ExpressiveMLM/data/mstro_with_dyn.pt',
                   device=torch.device('mps'),
                   perf_config_name='performance_with_dynamics',
                   n_embed=256,
                   max_seq_len=128,
                   n_layers=4,
                   n_heads=4,
                   dropout=0.1,
                   mlm_prob=0.15,
                   n_epochs=20
                   )
    run_mlm_test(test_settings=st)


if __name__ == '__main__':
    test_velocitymlm()
