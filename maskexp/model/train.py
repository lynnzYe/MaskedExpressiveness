import pickle

from maskexp.model.create_dataset import load_dataset
from maskexp.definitions import DATA_DIR, OUTPUT_DIR
from pathlib import Path
from maskexp.util.prepare_data import mask_perf_tokens
import time
import os
import torch
import torch.nn.functional as F
import tqdm
import note_seq
import matplotlib.pyplot as plt
from maskexp.model.bert import NanoBertMLM
from maskexp.magenta.models.performance_rnn import performance_model

MAX_SEQ_LEN = 128

NDEBUG = True


def decode_perf_logits(logits, decoder=None, idx=0):
    if decoder is None:
        raise ValueError("Decoder is required")
    pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1)[0, :].tolist()
    out = []
    for tk in pred_ids:
        out.append(decoder.decode_event(tk))
    return out


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_dir='checkpoint', name='checkpoint'):
    path = Path(save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, f'{save_dir}/{name}.pth')
    print(f"\x1B[34m[Info]\033[0m Checkpoint saved to {save_dir}/{name}.pth")


def print_perf_seq(perf_seq):
    outstr = ''
    for e in perf_seq:
        outstr += f'[{e.event_type}-{e.event_value}], '
    print(outstr)


def train_mlm(model, config, optimizer, n_epochs, train_dataloader, val_dataloader,
              mlm_prob=0.15, name='bert', device=torch.device('mps'), eval_epoch=5,
              save_output_dir=''):
    train_loss, val_loss = [], []
    tokenizer = config.encoder_decoder
    for epoch in range(n_epochs):
        train_start = time.time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            input_ids = batch[0]
            attention_mask = batch[1]

            # Mask tokens
            inputs, labels = mask_perf_tokens(input_ids, perf_config=config, mask_prob=mlm_prob,
                                              special_ids=(note_seq.PerformanceEvent.VELOCITY,))
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, preds = model(inputs, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if not NDEBUG:
                break

        avg_train_loss = total_loss / len(train_dataloader)
        train_end = time.time()
        print(
            f'Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss}, Training Time: {train_end - train_start}')
        train_loss.append(avg_train_loss)

        erange = epoch + 1 if NDEBUG else epoch
        if erange % eval_epoch == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for i_b, batch in enumerate(val_dataloader):
                    input_ids = batch[0]
                    attention_mask = batch[1]

                    # Mask tokens
                    inputs, labels = mask_perf_tokens(input_ids, perf_config=config, mask_prob=mlm_prob,
                                                      special_ids=(note_seq.PerformanceEvent.VELOCITY,))

                    inputs = inputs.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                    loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
                    total_val_loss += loss.item()

                    if i_b == len(val_dataloader) - 1:
                        decoded = [tokenizer.class_index_to_event(i, None) for i in inputs[0, :].tolist()]
                        print_perf_seq(decoded)
                        print_perf_seq(decode_perf_logits(logits, tokenizer._one_hot_encoding))

                    if not NDEBUG:
                        break

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {avg_val_loss}')
                val_loss.append(avg_val_loss)

                # if epoch % 5 == 0 or epoch == n_epochs - 1:
                save_checkpoint(model, optimizer, 0, train_loss, val_loss,
                                save_dir=f'{save_output_dir}/checkpoints',
                                name=name)

    return train_loss, val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_mlm_train(model_name='velocitymlm', device=torch.device('mps'),
                  save_dir='', data_dir=DATA_DIR + '/mstro_with_dyn.pt'):
    settings = {'model_name': model_name, 'save_dir': save_dir, 'device': device, 'data_dir': data_dir,
                'perf_config': 'performance_with_dynamics',
                'n_embed': 256,
                'max_seq_len': MAX_SEQ_LEN,
                'n_layers': 4,
                'n_heads': 4,
                'dropout': 0.1,
                'mlm_prob': 0.15, }
    config = performance_model.default_configs['performance_with_dynamics']

    train, val, test = load_dataset(data_dir)
    config = config

    model = NanoBertMLM(vocab_size=config.encoder_decoder.num_classes,
                        n_embed=256,
                        max_seq_len=MAX_SEQ_LEN,
                        n_layers=4,
                        n_heads=4,
                        dropout=0.1)
    print('n params:', count_parameters(model))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_loss, val_loss = train_mlm(model, config, optimizer, 20, train, val,
                                     mlm_prob=0.15, name=model_name, device=device, save_output_dir=save_dir)

    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Val Loss')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    path = Path(f'{save_dir}/logs')
    if not path.exists():
        path.mkdir(parents=True)
    plt.savefig(os.path.join(save_dir, 'logs', 'bertmlm-ans_losses.png'))


if __name__ == '__main__':
    run_mlm_train()
    pass
