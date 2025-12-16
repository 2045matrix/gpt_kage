# Transformer 架构完整实现指南

## 目录
1. [实现组件概览](#实现组件概览)
2. [基础组件实现](#基础组件实现)
3. [结构单元实现](#结构单元实现)
4. [完整模型实现](#完整模型实现)
5. [数据处理](#数据处理)
6. [训练流程](#训练流程)
7. [推理流程](#推理流程)
8. [整体总结](#整体总结)

---

## 实现组件概览

Transformer 架构的关键组件如下：

1. **ScaledDotProductAttention** - 缩放点积注意力机制
2. **MultiHeadAttention** - 多头自注意力机制
3. **PoswiseFeedForwardNet** - 逐位置前馈网络
4. **get_sin_enc_table** - 正弦位置编码表生成
5. **get_attn_pad_mask** - 填充掩码生成
6. **get_attn_subsequent_mask** - 后续掩码生成
7. **EncoderLayer** - 编码器层
8. **Encoder** - 完整编码器
9. **DecoderLayer** - 解码器层
10. **Decoder** - 完整解码器
11. **Transformer** - 顶层模型

---

## 基础组件实现

### 1. ScaledDotProductAttention（缩放点积注意力）

```python
import numpy as np
import torch
import torch.nn as nn

d_k = 64  # K(=Q)维度
d_v = 64  # V维度

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()        
    
    def forward(self, Q, K, V, attn_mask):
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v]
        # attn_mask [batch_size, n_heads, len_q, len_k]
        
        # 计算注意力分数（原始权重）
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores [batch_size, n_heads, len_q, len_k]
        
        # 使用注意力掩码，将attn_mask中值为1的位置的权重替换为极小值
        scores.masked_fill_(attn_mask, -1e9) 
        
        # 用softmax函数对注意力分数进行归一化
        weights = nn.Softmax(dim=-1)(scores) 
        # weights [batch_size, n_heads, len_q, len_k]
        
        # 计算上下文向量（注意力的输出）
        context = torch.matmul(weights, V) 
        # context [batch_size, n_heads, len_q, dim_v]
        
        return context, weights
```

**关键点**：
- 缩放因子 1/√d_k 防止点积过大导致梯度消失
- masked_fill_ 屏蔽无效位置，防止模型关注填充或未来信息
- softmax 将分数归一化为概率分布

---

### 2. MultiHeadAttention（多头自注意力）

```python
d_embedding = 512  # Embedding的维度
n_heads = 8        # Multi-Head Attention中头的个数
batch_size = 3     # 每一批的数据大小

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)  # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)  # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)  # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    
    def forward(self, Q, K, V, attn_mask): 
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        
        residual, batch_size = Q, Q.size(0)  # 保留残差连接
        
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        
        # 将注意力掩码复制到多头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask [batch_size, n_heads, len_q, len_k]
        
        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        
        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        # context [batch_size, len_q, n_heads * dim_v]
        
        # 用线性层把连接后的多头自注意力结果转换为原始嵌入维度
        output = self.linear(context) 
        # output [batch_size, len_q, embedding_dim]
        
        # 与输入(Q)进行残差连接，并进行层归一化后输出
        output = self.layer_norm(output + residual)
        # output [batch_size, len_q, embedding_dim]
        
        return output, weights
```

**关键点**：
- 线性投影将输入映射到每个头的维度空间
- transpose(1,2) 调整维度便于多头并行计算
- 多头拼接后再用线性层还原到 embedding 维度
- 残差连接和 LayerNorm 保证训练稳定性

**与 ScaledDotProductAttention 的关系**：
- MultiHeadAttention 内部调用 ScaledDotProductAttention
- MultiHeadAttention 负责多头的线性投影、reshape、拼接
- ScaledDotProductAttention 负责单头的注意力核心计算

---

### 3. PoswiseFeedForwardNet（逐位置前馈网络）

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # 定义一维卷积层1，用于将输入映射到更高维度
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 定义一维卷积层2，用于将输入映射回原始维度
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义层归一化
        self.layer_norm = nn.LayerNorm(d_embedding)
    
    def forward(self, inputs): 
        # inputs [batch_size, len_q, embedding_dim]
        
        residual = inputs  # 保留残差连接 
        
        # 在卷积层1后使用ReLU函数 
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) 
        # output [batch_size, d_ff, len_q]
        
        # 使用卷积层2进行降维 
        output = self.conv2(output).transpose(1, 2) 
        # output [batch_size, len_q, embedding_dim]
        
        # 与输入进行残差连接，并进行层归一化
        output = self.layer_norm(output + residual) 
        # output [batch_size, len_q, embedding_dim]
        
        return output
```

**关键点**：
- 两个 Conv1d（等价于两个线性层）进行升维和降维
- ReLU 提供非线性表达能力
- 残差连接和 LayerNorm 保证训练稳定

---

### 4. get_sin_enc_table（正弦位置编码）

```python
def get_sin_enc_table(n_position, embedding_dim):
    # n_position: 输入序列的最大长度
    # embedding_dim: 词嵌入向量的维度
    
    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))    
    
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle    
    
    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1 奇数维
    # sinusoid_table 的维度是 [n_position, embedding_dim]
    
    return torch.FloatTensor(sinusoid_table)
```

**公式**：
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**作用**：让模型感知序列中的相对位置关系，固定编码，不需训练

---

### 5. get_attn_pad_mask（填充掩码）

```python
def get_attn_pad_mask(seq_q, seq_k):
    # seq_q 的维度是 [batch_size, len_q]
    # seq_k 的维度是 [batch_size, len_k]
    
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # 生成布尔类型张量，<PAD>token的编码值为0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # pad_attn_mask 的维度是 [batch_size, 1, len_k]
    
    # 变形为与注意力分数相同形状的张量 
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    # pad_attn_mask 的维度是 [batch_size, len_q, len_k]
    
    return pad_attn_mask
```

**作用**：屏蔽填充位置，防止模型关注无效信息

---

### 6. get_attn_subsequent_mask（后续掩码）

```python
def get_attn_subsequent_mask(seq):
    # seq 的维度是 [batch_size, seq_len]
    
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 上三角矩阵，当前位置之后的都是1（需要掩码）
    
    return torch.from_numpy(subsequent_mask).bool()
```

**作用**：防止解码器在自注意力中"偷窥"未来信息（当前位置后的所有位置都被掩码）

---

## 结构单元实现

### 7. EncoderLayer（编码器层）

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()        
        self.enc_self_attn = MultiHeadAttention()     # 多头自注意力层        
        self.pos_ffn = PoswiseFeedForwardNet()        # 逐位置前馈网络层
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs 的维度是 [batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask 的维度是 [batch_size, seq_len, seq_len]
        
        # 将相同的Q, K, V输入多头自注意力层
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, enc_self_attn_mask)
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim] 
        # attn_weights 的维度是 [batch_size, n_heads, seq_len, seq_len]      
        
        # 将多头自注意力outputs输入逐位置前馈网络层
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim] 
        
        return enc_outputs, attn_weights
```

**结构**：自注意力 → 前馈网络（每个子层都含残差+LayerNorm）

---

### 8. DecoderLayer（解码器层）

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()        
        self.dec_self_attn = MultiHeadAttention()     # 多头自注意力层       
        self.dec_enc_attn = MultiHeadAttention()      # 编码器-解码器注意力层        
        self.pos_ffn = PoswiseFeedForwardNet()        # 逐位置前馈网络层
    
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs 的维度是 [batch_size, target_len, embedding_dim]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
        
        # 将相同的Q, K, V输入多头自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                        dec_inputs, dec_self_attn_mask)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attn 的维度是 [batch_size, n_heads, target_len, target_len]
        
        # 将解码器输出和编码器输出输入多头自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                      enc_outputs, dec_enc_attn_mask)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_enc_attn 的维度是 [batch_size, n_heads, target_len, source_len]
        
        # 输入逐位置前馈网络层
        dec_outputs = self.pos_ffn(dec_outputs)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        
        return dec_outputs, dec_self_attn, dec_enc_attn
```

**结构**：自注意力 → 互注意力 → 前馈网络

**关键差异**：
- 编码器每层只有自注意力（编码内部信息）
- 解码器每层有两种注意力：自注意力（目标内部） + 互注意力（源-目交互）

---

## 完整模型实现

### 9. Encoder（编码器）

```python
n_layers = 6  # Encoder的层数

class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()        
        self.src_emb = nn.Embedding(len(corpus.src_vocab), d_embedding)  # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained( \
          get_sin_enc_table(corpus.src_len+1, d_embedding), freeze=True)  # 位置嵌入层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 编码器层数
    
    def forward(self, enc_inputs):  
        # enc_inputs 的维度是 [batch_size, source_len]
        
        # 创建一个从1到source_len的位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        # pos_indices 的维度是 [1, source_len]
        
        # 对输入进行词嵌入和位置嵌入相加
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        
        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        # enc_self_attn_mask 的维度是 [batch_size, len_q, len_k]
        
        enc_self_attn_weights = []  # 初始化
        
        # 通过编码器层
        for layer in self.layers: 
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        # enc_self_attn_weights 是一个列表，每个元素的维度是[batch_size, n_heads, seq_len, seq_len]
        
        return enc_outputs, enc_self_attn_weights
```

**过程**：
1. 词嵌入和位置嵌入相加
2. 生成填充掩码
3. 依次通过 n 层 EncoderLayer

---

### 10. Decoder（解码器）

```python
n_layers = 6  # Decoder的层数

class Decoder(nn.Module):
    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab), d_embedding)  # 词嵌入层
        self.pos_emb = nn.Embedding.from_pretrained( \
           get_sin_enc_table(corpus.tgt_len+1, d_embedding), freeze=True)  # 位置嵌入层        
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # 层数
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs): 
        # dec_inputs 的维度是 [batch_size, target_len]
        # enc_inputs 的维度是 [batch_size, source_len]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        
        # 创建位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        # pos_indices 的维度是 [1, target_len]
        
        # 对输入进行词嵌入和位置嵌入相加
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        
        # 生成三种掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)           # 填充掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)         # 后续掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask \
                                       + dec_self_attn_subsequent_mask), 0) 
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)                # 解码器-编码器掩码
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
        
        dec_self_attns, dec_enc_attns = [], []  # 初始化
        
        # 通过解码器层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, 
                                               dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, source_len]
        
        return dec_outputs, dec_self_attns, dec_enc_attns
```

**关键点**：
- 生成三种掩码：填充掩码、后续掩码、互注意力掩码
- 填充掩码和后续掩码相加（逻辑或）作为自注意力掩码

---

### 11. Transformer（顶层模型）

```python
class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()        
        self.encoder = Encoder(corpus)                                    # 初始化编码器
        self.decoder = Decoder(corpus)                                    # 初始化解码器
        # 定义线性投影层，将解码器输出转换为目标词汇表大小的概率分布
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab), bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs 的维度是 [batch_size, source_seq_len]
        # dec_inputs 的维度是 [batch_size, target_seq_len]
        
        # 将输入传递给编码器
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # enc_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, src_seq_len, src_seq_len]
        
        # 将编码器输出、解码器输入和编码器输入传递给解码器
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        
        # 将解码器输出传递给投影层，生成目标词汇表大小的概率分布
        dec_logits = self.projection(dec_outputs)  
        # dec_logits 的维度是 [batch_size, tgt_seq_len, tgt_vocab_size]
        
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
```

---

## 数据处理

### TranslationCorpus（翻译语料库）

```python
from collections import Counter

class TranslationCorpus:
    def __init__(self, sentences):
        self.sentences = sentences
        # 计算源语言和目标语言的最大句子长度
        self.src_len = max(len(sentence[0].split()) for sentence in sentences) + 1
        self.tgt_len = max(len(sentence[1].split()) for sentence in sentences) + 2
        # 创建源语言和目标语言的词汇表
        self.src_vocab, self.tgt_vocab = self.create_vocabularies()
        # 创建索引到单词的映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}
    
    def create_vocabularies(self):
        # 统计源语言和目标语言的单词频率
        src_counter = Counter(word for sentence in self.sentences for word in sentence[0].split())
        tgt_counter = Counter(word for sentence in self.sentences for word in sentence[1].split())        
        # 创建词汇表
        src_vocab = {'<pad>': 0, **{word: i+1 for i, word in enumerate(src_counter)}}
        tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 
                     **{word: i+3 for i, word in enumerate(tgt_counter)}}        
        return src_vocab, tgt_vocab
    
    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch, target_batch = [], [], []
        # 随机选择句子索引
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        
        for index in sentence_indices:
            src_sentence, tgt_sentence = self.sentences[index]
            # 将源语言和目标语言的句子转换为索引序列
            src_seq = [self.src_vocab[word] for word in src_sentence.split()]
            tgt_seq = [self.tgt_vocab['<sos>']] + [self.tgt_vocab[word] \
                         for word in tgt_sentence.split()] + [self.tgt_vocab['<eos>']]            
            # 对序列进行填充
            src_seq += [self.src_vocab['<pad>']] * (self.src_len - len(src_seq))
            tgt_seq += [self.tgt_vocab['<pad>']] * (self.tgt_len - len(tgt_seq))            
            # 将处理好的序列添加到批次中
            input_batch.append(src_seq)
            output_batch.append([self.tgt_vocab['<sos>']] + ([self.tgt_vocab['<pad>']] * \
                                    (self.tgt_len - 2)) if test_batch else tgt_seq[:-1])
            target_batch.append(tgt_seq[1:])        
        
        # 将批次转换为LongTensor类型
        input_batch = torch.LongTensor(input_batch)
        output_batch = torch.LongTensor(output_batch)
        target_batch = torch.LongTensor(target_batch)            
        return input_batch, output_batch, target_batch

# 示例数据
sentences = [
    ['咖哥 喜欢 小冰', 'KaGe likes XiaoBing'],
    ['我 爱 学习 人工智能', 'I love studying AI'],
    ['深度学习 改变 世界', 'DL changed the world'],
    ['自然语言处理 很 强大', 'NLP is powerful'],
    ['神经网络 非常 复杂', 'Neural-networks are complex']
]

corpus = TranslationCorpus(sentences)
```

---

## 训练流程

```python
import torch.optim as optim

# 创建模型实例
model = Transformer(corpus)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练参数
epochs = 100
batch_size = 3

# 训练循环
for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零
    
    # 创建训练数据
    enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size)
    
    # 前向传播
    outputs, _, _, _ = model(enc_inputs, dec_inputs)
    
    # 计算损失
    loss = criterion(outputs.view(-1, len(corpus.tgt_vocab)), target_batch.view(-1))
    
    # 打印损失
    if (epoch + 1) % 1 == 0:
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
```

**训练日志示例**：
```
Epoch: 0020 cost = 0.019715
Epoch: 0040 cost = 0.003158
Epoch: 0060 cost = 0.001584
Epoch: 0080 cost = 0.001170
Epoch: 0100 cost = 0.000925
```

---

## 推理流程

```python
# 测试：生成一个大小为1的批次，目标语言序列只包含<sos>
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True) 

# 进行翻译
predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

# 将预测结果维度重塑
predict = predict.view(-1, len(corpus.tgt_vocab))

# 找到每个位置概率最大的单词的索引
predict = predict.data.max(1, keepdim=True)[1]

# 解码预测的输出
translated_sentence = [corpus.tgt_idx2word[idx.item()] for idx in predict.squeeze()]

# 解码输入的源语言句子
input_sentence = ' '.join([corpus.src_idx2word[idx.item()] for idx in enc_inputs[0]])

# 打印结果
print(input_sentence, '->', translated_sentence)
```

**注意**：当前实现只生成一步预测。要实现完整的自回归翻译，需要循环生成，每次将上一步的预测结果作为下一步的输入，直到生成<eos>或达到最大长度。

---

## 整体总结

### 实现层级关系

```
基础组件层
├── ScaledDotProductAttention（单头注意力）
├── MultiHeadAttention（多头自注意力）
├── PoswiseFeedForwardNet（前馈网络）
├── get_sin_enc_table（位置编码）
├── get_attn_pad_mask（填充掩码）
└── get_attn_subsequent_mask（后续掩码）

结构单元层
├── EncoderLayer（编码器基本单元）
└── DecoderLayer（解码器基本单元）

完整模型层
├── Encoder（编码器）
├── Decoder（解码器）
└── Transformer（顶层模型）

应用层
├── TranslationCorpus（数据处理）
├── 训练流程
└── 推理流程
```

### Transformer架构示意图

```
           +-------------------+
           |   输入序列 Input   |
           +-------------------+
                     |
                     v
           +-------------------+
           |   词嵌入+位置编码  |
           +-------------------+
                     |
                     v
           +-------------------+
           |    编码器堆叠N层   |
           |  (Encoder Layers)  |
           +-------------------+
                     |
           编码器输出 Enc Output
                     |
                     v
           +-------------------+
           |   目标序列 Target  |
           +-------------------+
                     |
                     v
           +-------------------+
           |   词嵌入+位置编码  |
           +-------------------+
                     |
                     v
           +-------------------+
           |   解码器堆叠N层    |
           | (Decoder Layers)   |
           +-------------------+
                     |
                     v
           +-------------------+
           |   线性+Softmax     |
           +-------------------+
                     |
                     v
           +-------------------+
           |   输出概率分布     |
           +-------------------+
```

### 关键设计要点

1. **多头自注意力**：并行计算多个头，增强模型表达能力
2. **前馈网络**：两层线性层 + ReLU，增加非线性
3. **位置编码**：正弦/余弦位置编码，固定不训练
4. **掩码机制**：
   - 填充掩码：屏蔽<pad>
   - 后续掩码：防止解码器看到未来信息
5. **残差连接+层归一化**：每个子层都有，保证梯度流动和训练稳定
6. **编码器-解码器注意力**：实现源-目信息交互的关键

### 编码器 vs 解码器

| 特性 | 编码器 | 解码器 |
|------|--------|--------|
| 自注意力个数 | 1 | 1 |
| 互注意力 | 无 | 有 |
| 掩码 | 仅填充掩码 | 填充 + 后续掩码 |
| Q/K/V来源 | 自身 | 自身+编码器输出 |

---

## 完整工作流

1. **数据准备**：词汇表构建、句子编码、填充、批处理
2. **模型初始化**：创建 Transformer 实例
3. **训练**：前向传播 → 计算损失 → 反向传播 → 参数更新
4. **推理**：编码源序列 → 解码生成目标序列
5. **评估**：对比翻译结果和目标翻译

这样就完成了从组件实现、模型构建、数据处理到训练推理的完整 Transformer 架构！
