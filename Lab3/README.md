# regression-house-sale-price-prediction-challenge-310706038

# 作法說明

## 1.資料處理
首先針對training data的文字進行處理，因為training data是有經過特別的分詞，但testing data是一整個句子。  
為了使得兩者資料集一致，因此將training data 的分詞重處理，採用一個字一個字進行分詞，並且將原來的空格也處理掉。  
中文與台語羅馬的分詞分開處理。
```python
def isEng(word):
    ascii = ord(word)
    if (ascii >= 65 and ascii <= 90) or (ascii >= 97 and ascii <= 122):
        return True
    return False

def zh_single_word_tokenize(string):
    single_word_list = []
    eng_str = ""
    for s in string:
        if ord(s) == 32: continue #space
        if isEng(s): eng_str += s
        else:
            if len(eng_str) > 0:
                single_word_list.append(eng_str)
                eng_str = ""
            single_word_list.append(s)
        
    return single_word_list


def tl_single_word_tokenize(string):
    single_word_list = []
    string = string.split()
    for s in string:
        rm_dil = s.split("-")
        for c in rm_dil: single_word_list.append(c)

    return single_word_list
```

## 2. 詞表建立
將資料的分詞完成後，需要建立詞表對應，將字轉換為向量，並且將字與向量的對應關係記錄起來，建立一詞表。  
其中在詞表加入一些特別的詞定義，像是unkown, paddingbeginning of sentence, end of sentence等。

- BOS（Beginning of Sentence）： 在翻譯任務中，指示句子的開始。它有助於模型在生成目標語言的翻譯時明確句子的起始。
- EOS（End of Sentence）： 表示句子的結束。在生成階段，當模型預測到EOS標記時，它知道句子已經生成完畢。
- UNK（Unknown）： 用於表示模型在訓練過程中未見過的詞彙。當模型在翻譯或生成文本時遇到未知詞彙時，可以使用UNK標記代替，避免因為未知詞彙而導致輸出錯誤。
- PAD（Padding）： 在序列的短於最大長度的部分進行填充。在批處理中，為了方便處理不同長度的序列，通常使用PAD標記對短序列進行填充，使它們具有相同的長度。這有助於加速訓練過程，同時也對模型的輸入進行規範化。

```python
# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
vocab_transform["zh"] = build_vocab_from_iterator(zh_list,min_freq=1,specials=special_symbols,special_first=True)
vocab_transform["tl"] = build_vocab_from_iterator(tl_list,min_freq=1,specials=special_symbols,special_first=True)
vocab_transform["zh"].set_default_index(UNK_IDX)
vocab_transform["tl"].set_default_index(UNK_IDX)
```


## 3. 模型設定

### seq2seq transformer
此次翻譯採用seq2seq transformer model。  
其中的 PositionalEncoding，幫助word embedding添加位置編碼。  
TokenEmbedding，輸入索引的張量轉換為相應的標記嵌入。

```python

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
```

### mask
而在seq2seq transformer的模型中需要遮罩，為的是在decoder在翻譯時，只能使用前面的字詞，不能使用到後面的字詞，所以需要使用遮罩將未來的字詞遮住。
```python
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
```

### parameter setiing
```python
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
```


## 資料轉換
最後需要將資料進行轉換，利用先前建立的詞向量表，將data的文字都轉換為向量，並且在每一句子的開頭與結尾，分別加入bos, eos的向量，因為每個句子長度不一，需要進行padding，資料才可以形成batch。
```python
from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(#token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        # print(src_sample)
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
```


# 訓練與驗證loss圖
共訓練20個epoch，可以看到loss的部分在訓練與驗證的部分都有收斂。

![訓練驗證loss圖](https://github.com/Machine-Learning-NYCU/chinese-to-tailo-neural-machine-translation-310706038/blob/main/plot/learning_curve.png)


# 改進與心得討論
## 改進

1. 或許可以採用不同的切割詞的方式進行切割，再訓練

2. 調整超參數，像是attention head等。

## 心得討論
此次kaggle競賽讓我學習到如何處理一NLP相關的任務，透過分詞，建立詞向量，將文字轉換成機器看得懂的形式，再使用transformer model進行訓練，並且也更加瞭解到transformer model與self.attention的機制與概念。
