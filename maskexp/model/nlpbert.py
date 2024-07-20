import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizerFast
from datasets import load_dataset
import matplotlib.pyplot as plt
import time
import os

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
device = torch.device('mps')
max_length = 128
batch_size = 32


def bert_encode_text(texts, subset=None):
    input_ids = []
    attention_masks = []
    encoded_dict = tokenizer.batch_encode_plus(texts, add_special_tokens=True, max_length=max_length, truncation=True,
                                               padding='max_length', return_attention_mask=True,
                                               return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    tokens = tokenizer.tokenize(tokenizer.decode(input_ids[0]))
    if subset is None:
        return input_ids, attention_masks, tokens
    return input_ids[:subset], attention_masks[:subset], tokens


def prepare_data():
    # ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    ds = load_dataset("stanfordnlp/imdb")
    print(ds)

    train_texts = ds['train']['text']
    val_texts = ds['test']['text']

    # train_ids, train_msks, _ = bert_encode_text(train_texts[:10000])
    # val_ids, val_msks, _ = bert_encode_text(val_texts[:5000])

    train_ids, train_msks, _ = bert_encode_text(train_texts, subset=None)
    val_ids, val_msks, _ = bert_encode_text(val_texts, subset=None)

    # train_labels = torch.tensor(ds['train']['label']).to(device)
    # val_labels = torch.tensor(ds['test']['label'][:5000]).to(device)

    train_labels = torch.zeros(len(train_ids)).to(device)  # dummy
    val_labels = torch.zeros(len(val_ids)).to(device)  # dummy

    train_dataset = TensorDataset(train_ids, train_msks, train_labels)
    val_dataset = TensorDataset(val_ids, val_msks, val_labels)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)

    return train_dataloader, val_dataloader


class BertEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, n_embed=3, max_seq_len=16):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.word_embeddings = torch.nn.Embedding(vocab_size, n_embed)
        self.pos_embeddings = torch.nn.Embedding(max_seq_len, n_embed)

        self.layer_norm = torch.nn.LayerNorm(n_embed, eps=1e-12, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)

        words_embeddings = self.word_embeddings(x)
        position_embeddings = self.pos_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertAttentionHead(torch.nn.Module):
    """
    A single attention head in MultiHeaded Self Attention layer.
    The idea is identical to the original paper ("Attention is all you need"),
    however instead of implementing multiple heads to be evaluated in parallel we matrix multiplication,
    separated in a distinct class for easier and clearer interpretability
    """

    def __init__(self, head_size, dropout=0.1, n_embed=3):
        super().__init__()

        self.query = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.key = torch.nn.Linear(in_features=n_embed, out_features=head_size)
        self.values = torch.nn.Linear(in_features=n_embed, out_features=head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # B, Seq_len, N_embed
        B, seq_len, n_embed = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(n_embed)  # (B, Seq_len, Seq_len)
        weights = weights.masked_fill(mask == 0, -1e9)  # mask out not attended tokens

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        context = scores @ v

        return context


class BertSelfAttention(torch.nn.Module):
    """
    MultiHeaded Self-Attention mechanism as described in "Attention is all you need"
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        head_size = n_embed // n_heads
        n_heads = n_heads
        self.heads = torch.nn.ModuleList([BertAttentionHead(head_size, dropout, n_embed) for _ in range(n_heads)])
        self.proj = torch.nn.Linear(head_size * n_heads, n_embed)  # project from multiple heads to the single space
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        context = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        proj = self.proj(context)
        out = self.dropout(proj)

        return out


class FeedForward(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.ffwd = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.GELU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.ffwd(x)

        return out


class BertLayer(torch.nn.Module):
    """
    Single layer of BERT transformer model
    """

    def __init__(self, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        # unlike in the original paper, today in transformers it is more common to apply layer norm before other layers
        # this idea is borrowed from Andrej Karpathy's series on transformers implementation
        self.layer_norm1 = torch.nn.LayerNorm(n_embed)
        self.self_attention = BertSelfAttention(n_heads, dropout, n_embed)

        self.layer_norm2 = torch.nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(dropout, n_embed)

    def forward(self, x, mask):
        x = self.layer_norm1(x)
        x = x + self.self_attention(x, mask)

        x = self.layer_norm2(x)
        out = x + self.feed_forward(x)

        return out


class BertEncoder(torch.nn.Module):
    def __init__(self, n_layers=2, n_heads=1, dropout=0.1, n_embed=3):
        super().__init__()

        self.layers = torch.nn.ModuleList([BertLayer(n_heads, dropout, n_embed) for _ in range(n_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertPooler(torch.nn.Module):
    def __init__(self, dropout=0.1, n_embed=3):
        super().__init__()

        self.dense = torch.nn.Linear(in_features=n_embed, out_features=n_embed)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        pooled = self.dense(x)
        out = self.activation(pooled)

        return out


class NanoBERT(torch.nn.Module):
    """
    NanoBERT is a almost an exact copy of a transformer decoder part described in the paper "Attention is all you need"
    This is a base model that can be used for various purposes such as Masked Language Modelling, Classification,
    Or any other kind of NLP tasks.
    This implementation does not cover the Seq2Seq problem, but can be easily extended to that.
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        """

        :param vocab_size: size of the vocabulary that tokenizer is using
        :param n_layers: number of BERT layer in the model (default=2)
        :param n_heads: number of heads in the MultiHeaded Self Attention Mechanism (default=1)
        :param dropout: hidden dropout of the BERT model (default=0.1)
        :param n_embed: hidden embeddings dimensionality (default=3)
        :param max_seq_len: max length of the input sequence (default=16)
        """
        super().__init__()

        self.embedding = BertEmbeddings(vocab_size, n_embed, max_seq_len)

        self.encoder = BertEncoder(n_layers, n_heads, dropout, n_embed)

        self.pooler = BertPooler(dropout, n_embed)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        emb_output = self.embedding(input_ids)

        # mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1)

        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = attention_mask.repeat(1, input_ids.size(1), 1)

        # encoded = self.encoder(emb_output, mask)
        encoded = self.encoder(emb_output, attention_mask)

        pooled = self.pooler(encoded)
        return pooled


class NanoBertMLM(nn.Module):
    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16):
        super().__init__()
        self.bert = NanoBERT(vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
                             n_embed=n_embed, max_seq_len=max_seq_len)
        self.cls = nn.Sequential(
            nn.Linear(in_features=n_embed, out_features=n_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=n_embed, out_features=vocab_size)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prediction_scores.view(-1, self.cls[-1].out_features), labels.view(-1))

        return loss, prediction_scores


class NanoBertForClassification(torch.nn.Module):
    """
    This is a wrapper on the base NanoBERT that is used for classification task
    One can use this as an example of how to extend and apply nano-BERT to similar custom tasks
    This layer simply adds one additional dense layer for classification
    """

    def __init__(self, vocab_size, n_layers=2, n_heads=1, dropout=0.1, n_embed=3, max_seq_len=16, n_classes=2):
        super().__init__()
        self.nano_bert = NanoBERT(vocab_size, n_layers, n_heads, dropout, n_embed, max_seq_len)

        self.classifier = torch.nn.Linear(in_features=n_embed, out_features=n_classes)

    def forward(self, input_ids):
        embeddings = self.nano_bert(input_ids)

        logits = self.classifier(embeddings)
        return logits


def mask_tokens(inputs, tokenizer, mlm_prob=0.15):
    labels = inputs.clone()
    prob_matrix = torch.full(labels.shape, mlm_prob)
    special_token_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                          labels.tolist()]
    prob_matrix.masked_fill_(torch.tensor(special_token_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    # Replace with [MASK] 80%
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Random work 10%
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def decode_mlm_logits(logits):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
    # print(pred_ids.tolist()[0])
    print(tokenizer.decode(pred_ids[0].tolist()[:100]))


def save_checkpoint(model, optimizer, epoch, loss, path='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def train_mlm(model, optimizer, n_epochs, train_dataloader, val_dataloader, mlm_prob=0.15, name='bert'):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_loss, val_loss = [], []

    for epoch in range(n_epochs):
        train_start = time.time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            input_ids = batch[0]
            attention_mask = batch[1]

            # Mask tokens
            inputs, labels = mask_tokens(input_ids, tokenizer=tokenizer, mlm_prob=mlm_prob)
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, preds = model(inputs, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        train_end = time.time()
        print(
            f'Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss}, Training Time: {train_end - train_start}')
        train_loss.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i_b, batch in enumerate(val_dataloader):
                input_ids = batch[0]
                attention_mask = batch[1]

                # Mask tokens
                inputs, labels = mask_tokens(input_ids, tokenizer)

                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
                total_val_loss += loss.item()

                if i_b == len(val_dataloader) - 1:
                    print("input:", tokenizer.decode(inputs[0]))
                    decode_mlm_logits(logits)

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {avg_val_loss}')
            val_loss.append(avg_val_loss)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            save_checkpoint(model, optimizer, epoch, val_loss, path=f'checkpoints/{name}.pth')

    return train_loss, val_loss


def run_mlm_train(model_name='bertmlm'):
    train, val = prepare_data()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = NanoBertMLM(vocab_size=len(tokenizer.get_vocab().keys()),
                        n_embed=256,
                        max_seq_len=max_length,
                        n_layers=4,
                        n_heads=4,
                        dropout=0.1)
    print('n params:', count_parameters(model))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_loss, val_loss = train_mlm(model, optimizer, 20, train, val, mlm_prob=0.15, name=model_name)
    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Val Loss')
    plt.title('Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('logs', 'bertmlm-ans_losses.png'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    run_mlm_train(model_name='bertmlm-ans')
