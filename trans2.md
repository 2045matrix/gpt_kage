# Transformer 架构：从理论到实践完整指南

## 目录
1. [基础概念](#基础概念)
2. [注意力机制基础](#注意力机制基础)
3. [Query、Key、Value 概念](#querykeyvalue-概念)
4. [交互注意力 vs 自注意力](#交互注意力-vs-自注意力)
5. [多头自注意力](#多头自注意力)
6. [实现组件概览](#实现组件概览)
7. [基础组件实现](#基础组件实现)
8. [结构单元实现](#结构单元实现)
9. [完整模型实现](#完整模型实现)
10. [数据处理](#数据处理)
11. [训练流程](#训练流程)
12. [推理流程](#推理流程)
13. [整体总结](#整体总结)

---

## 基础概念

### 1. 张量与序列表示

- **x1 = torch.randn(2, 3, 4)**：形状(batch_size, seq_len1, feature_dim)
- **x2 = torch.randn(2, 5, 4)**：形状(batch_size, seq_len2, feature_dim)

**张量数据示例：**
- x1：表示解码器的隐藏状态序列，形状 [batch_size=2, seq_len1=3, feature_dim=4]
  - 每个样本有3个时间步，每个时间步有4个维度特征
  - 例如：x1[0, 0, :] = [0.2, -0.5, 1.3, 0.8] 表示第一个样本第一个时间步的特征向量
- x2：表示编码器的隐藏状态序列，形状 [batch_size=2, seq_len2=5, feature_dim=4]
  - 每个样本有5个时间步，每个时间步有4个维度特征
  - 例如：x2[0, 0, :] = [0.1, 0.3, -0.2, 0.9] 表示第一个样本第一个时间步的特征向量

### 2. Seq2Seq 架构中的 x1 和 x2

在Seq2Seq架构中，点积注意力通常用于将编码器的隐藏状态与解码器的隐藏状态联系起来：
- **x1**：解码器在各个时间步的隐藏状态 [batch_size, seq_len1, feature_dim]
- **x2**：编码器在各个时间步的隐藏状态 [batch_size, seq_len2, feature_dim]

---

## 注意力机制基础

### 1. 基础注意力（Attention）

```python
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    def forward(self, decoder_context, encoder_context):
        # 计算点积分数
        scores = torch.matmul(decoder_context, encoder_context.transpose(-2, -1))
        # 归一化分数
        attn_weights = nn.functional.softmax(scores, dim=-1)
        # 加权求和
        context = torch.matmul(attn_weights, encoder_context)
        return context, attn_weights
```

### 2. 解码器与注意力（DecoderWithAttention）

```python
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention()
        self.out = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, dec_input, hidden, enc_output):
        embedded = self.embedding(dec_input)
        rnn_output, hidden = self.rnn(embedded, hidden)
        context, attn_weights = self.attention(rnn_output, enc_output)
        dec_output = torch.cat((rnn_output, context), dim=-1)
        dec_output = self.out(dec_output)
        return dec_output, hidden, attn_weights
```

在这里：
- x1 = rnn_output（解码器隐藏状态）
- x2 = enc_output（编码器隐藏状态）

### 3. Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, encoder_input, hidden, decoder_input):
        encoder_output, encoder_hidden = self.encoder(encoder_input, hidden)
        decoder_hidden = encoder_hidden
        decoder_output, _, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_output)
        return decoder_output, attn_weights
```

---

## Query、Key、Value 概念

### 1. 三个关键部分

- **查询（Query）**：当前需要处理的信息。模型根据查询向量在输入序列中查找相关信息。
- **键（Key）**：来自输入序列的一组表示。用于根据查询向量计算注意力权重。
- **值（Value）**：来自输入序列的一组表示。用于根据注意力权重计算加权和。

### 2. Seq2Seq 中的 Q、K、V 映射

- x1（解码器隐藏状态）→ Query（q）
- x2（编码器隐藏状态）→ Key（k）和 Value（v）

在基础点积注意力中，k 和 v 通常使用同一个张量（x2）。

### 3. 注意力计算过程

1. 用查询向量与各个键向量计算相似性（点积），得到注意力分数
2. 对分数做softmax归一化，得到权重
3. 用权重对所有值向量加权求和，得到输出向量
4. 输出向量蕴含输入序列中与查询最相关的信息

---

## 交互注意力 vs 自注意力

### 1. 交互注意力（Encoder-Decoder Attention）

**特点：**
- Query 来自解码器（x1）
- Key/Value 来自编码器（x2）
- 解码器每一步都关注编码器输出的所有部分

**应用：**
- 传统Seq2Seq模型
- 机器翻译、文本摘要等任务

**数据处理说明：**
- x1（解码器隐藏状态）在每个时间步都关注x2（编码器隐藏状态）的所有位置
- 通过缩放点积注意力机制：`attention = softmax(x1 @ x2^T / sqrt(d_k)) @ x2`
- 计算x1每个位置对x2各位置的关联强度，并加权融合x2的信息
- 最后得到融合了编码器信息的解码器表示，用于后续的输出预测

### 2. 自注意力（Self-Attention）

**特点：**
- Query、Key、Value 都来自同一个序列
- x1 = x2 = x（同一张量）
- 序列内部每个位置都能关注其它所有位置

**应用：**
- Transformer编码器和解码器
- ChatGPT等大规模语言模型
- 全局信息融合

**代码示例：**
```python
x = torch.randn(2, 3, 4)
raw_weights = torch.bmm(x, x.transpose(1, 2))
attn_weights = F.softmax(raw_weights, dim=2)
attn_outputs = torch.bmm(attn_weights, x)
```

### 3. 两者的核心区别

| 特性 | 交互注意力 | 自注意力 |
|-----|---------|--------|
| Q/K/V来源 | 不同序列（解码器/编码器） | 同一序列 |
| 适用场景 | Seq2Seq | Transformer |
| 信息交互 | 输入→输出 | 序列内部 |
| 应用模型 | 传统神经机器翻译 | ChatGPT、BERT等 |

---

## 多头自注意力

### 1. 计算过程

1. **初始化**：设定多个头，每个头有独立的线性变换权重
2. **线性变换**：对Q、K、V进行多次线性变换，每次使用不同的权重矩阵
3. **缩放点积注意力**：每组Q、K、V独立计算注意力，每个头生成一个加权输出
4. **合并**：将所有头的输出拼接，再进行一次线性变换

### 2. 优势

- 通过同时学习多个子空间的特征，提高模型捕捉长距离依赖的能力
- 能在不同的语义层次上并行关注信息
- 增强模型表达能力

### 3. 多头注意力的作用

在Transformer自注意力机制中，多头的优势：
- **多个投影空间**：每个头用不同的线性变换 W_Q、W_K、W_V，将输入投影到不同的特征空间
- **并行计算**：n_heads个头同时计算，每个头关注不同的语义和特征模式
- **组合融合**：将所有头的输出拼接，再通过线性变换整合，得到更丰富的表示

### 4. 数据流概览

- **输入**：x（或x1、x2） 形状 [batch_size, seq_len, d_embedding]
- **Q、K、V投影**：每个头独立投影到 [batch_size, n_heads, seq_len, d_k]
- **注意力计算**：scores = Q @ K^T / sqrt(d_k)，weights = softmax(scores)，context = weights @ V
- **多头拼接**：[batch_size, seq_len, n_heads * d_v]
- **输出投影**：通过线性层还原到 [batch_size, seq_len, d_embedding]

---

## 位置编码与掩码机制

### 位置编码（Positional Encoding）

在Transformer中，位置编码的目的是为模型提供序列中词的位置信息。由于自注意力机制是全局的，无法天然理解词的顺序，因此需要显式加入位置信息。

位置编码通常使用正弦和余弦函数，并与词嵌入向量相加，补充序列顺序信息。

### 注意力掩码（Attention Mask）

**1. 填充注意力掩码（Padding Attention Mask）**
- 作用：屏蔽掉输入序列中填充（padding）的位置，防止模型关注无效信息。
- 应用场景：Transformer编码器、解码器的所有注意力层。

**2. 前瞻/后续注意力掩码（Look-ahead/Subsequent Attention Mask）**
- 作用：在自回归生成时，防止模型看到未来的信息，保证生成的因果性。
- 应用场景：Transformer解码器的自注意力层。

**总结：**
- 编码器只需填充掩码。
- 解码器既需填充掩码，也需前瞻掩码。

---

## Transformer 编码器内部结构

### 编码器单层结构（伪代码）
```python
# 输入: x  (shape: [batch, seq_len, d_model])

# 1. 多头自注意力
attn_output = MultiHeadSelfAttention(x)  # [batch, seq_len, d_model]

# 2. 残差连接 + 层归一化
x1 = LayerNorm(x + attn_output)          # 第一次Add & Norm

# 3. 前馈神经网络
ffn_output = FeedForwardNetwork(x1)      # [batch, seq_len, d_model]

# 4. 残差连接 + 层归一化
output = LayerNorm(x1 + ffn_output)      # 第二次Add & Norm
```

### FFN 内部结构
```python
def FeedForwardNetwork(x):
    y = Linear1(x)         # 升维（如512→2048）
    y = ReLU(y)            # 非线性激活
    y = Linear2(y)         # 降维（如2048→512）
    return y
```

- 激活函数作用：引入非线性，使网络能拟合更复杂的函数关系。
- ReLU：简单高效，当x>0时输出x，否则输出0。

---

## Transformer 解码器内部结构

### 解码器单层结构
1. 多头自注意力（Masked Multi-Head Self-Attention）
   - 只能关注已生成的目标序列（用掩码防止"看未来"）
2. 编码器-解码器注意力（Encoder-Decoder Attention）
   - Query来自解码器自注意力输出，Key/Value来自编码器输出
3. 前馈神经网络（FFN）
4. 每个子层都有残差连接+层归一化

### 解码器输入与目标序列的关系

训练时（使用x1表示目标序列）：
- decoder_input: <sos> x1_token1 x1_token2 x1_token3 x1_token4  
- Target:        x1_token1 x1_token2 x1_token3 x1_token4 <eos>
  
其中x1_token代表目标序列中的各个词，每个词都被编码为整数索引，然后转换为嵌入向量

作用：模型每一步都用当前输入预测"下一个词"，最终生成完整目标序列。

### 训练与推理的区别
- 训练（教师强制）：用真实目标序列作为解码器输入
- 推理（自回归生成）：用模型自己已生成的词作为当前输入

### Transformer 输出层

1. 线性层：将解码器最后输出映射到词汇表大小的空间，得到每个词的分数。
2. softmax层：将分数转为概率分布（所有概率和为1）。
3. 最后得到每个位置上所有词的概率分布，可用于生成或分类。

---

## 相关概念补充

### RNN、LSTM、GRU简述

三者都是处理序列数据的循环神经网络：

- **RNN**（Recurrent Neural Network）：基础版本，每步输出依赖当前输入和上一步隐藏状态。容易梯度消失/爆炸，难以捕捉长距离依赖。

- **LSTM**（Long Short-Term Memory）：引入"门控机制"和"细胞状态"，能有效记忆和遗忘信息，解决了RNN的长距离依赖问题。

- **GRU**（Gated Recurrent Unit）：结构比LSTM更简单，只有重置门和更新门，也能捕捉长距离依赖，计算更高效。

本质：通过"循环"结构把前面信息传递到后面，实现序列建模。

### Skip-gram、Seq2Seq、Transformer对比

| 特性 | Skip-gram | Seq2Seq | Transformer |
|-----|-----------|---------|------------|
| 任务 | 词向量训练 | 序列到序列 | 通用序列建模 |
| 结构 | 单层前馈网络 | RNN/LSTM/GRU编码器+解码器 | 全自注意力+前馈 |
| 编码器-解码器 | 无 | 有 | 有 |
| 计算方式 | 单步 | 串行（逐步） | 并行（全局） |
| 长距离依赖 | 无（静态） | 有但受限 | 强大 |
| 并行能力 | 无 | 无 | 全并行 |
| 幻觉风险 | 低 | 低 | 较高 |
| 易扩展性 | 低 | 中 | 高（ChatGPT等） |

### Transformer vs Seq2Seq的关键区别

**Transformer并行，Seq2Seq串行：**
- Transformer：所有位置同时计算（自注意力），训练推理速度快，但容易产生幻觉。
- Seq2Seq：每步依赖前一步（循环结构），必须按顺序处理，严格约束输出，幻觉较少。

**任务定义：**
- Seq2Seq：输入和输出都是序列（如机器翻译、文本摘要、对话生成）
- 非序列任务：输入或输出不是序列（如文本分类、图片识别）

---

## Transformer 实现组件概览

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
        # 输入维度分析 (对应x2的源头):
        # enc_inputs [batch_size=3, source_len=5]  # 原始token索引序列
        # 例如: [[1, 2, 3, 0, 0], [2, 3, 4, 5, 0], [1, 3, 0, 0, 0]]  # 5个token位置
        
        # 创建一个从1到source_len的位置索引序列
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        # pos_indices [1, source_len=5]  # 位置: [1, 2, 3, 4, 5]
        
        # 对输入进行词嵌入和位置嵌入相加
        # self.src_emb(enc_inputs) [batch_size=3, source_len=5, d_embedding=512]
        # self.pos_emb(pos_indices) [1, source_len=5, d_embedding=512]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        
        # 编码器输出维度 (对应x2 - 编码器隐藏状态):
        # enc_outputs [batch_size=3, source_len=5, d_embedding=512]
        # 这就是Seq2Seq中的x2 (编码器在各个时间步的隐藏状态)
        
        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) 
        # enc_self_attn_mask [batch_size=3, source_len=5, source_len=5]
        
        enc_self_attn_weights = []  # 初始化
        
        # 通过编码器层 (N=6层堆叠)
        for layer in self.layers: 
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            # 每层输入输出维度保持不变:
            # enc_outputs [batch_size=3, source_len=5, d_embedding=512]
            # enc_self_attn_weight [batch_size=3, n_heads=8, source_len=5, source_len=5]
            
            enc_self_attn_weights.append(enc_self_attn_weight)
        
        # 最终编码器输出 (完整的x2):
        # enc_outputs [batch_size=3, source_len=5, d_embedding=512]
        # 包含源序列的完整上下文信息，将被传递给解码器
        
        return enc_outputs, enc_self_attn_weights
```

**数据流分析**：
- **输入**：enc_inputs [batch_size, source_len] - 源序列token索引
- **输出**：enc_outputs [batch_size, source_len, d_embedding] - **x2** (编码器隐藏状态)
- **作用**：将源序列转化为编码器上下文向量，供解码器的互注意力使用

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
        # 输入维度分析 (对应x1和x2的交互):
        # dec_inputs [batch_size=3, target_len=8]      # 目标序列token索引(带<sos>)
        #   例如: [[1, 2, 3, 4, 0, 0, 0, 0], ...]     # <sos>=1, 目标词汇
        # enc_inputs [batch_size=3, source_len=5]      # 源序列(用于生成掩码)
        # enc_outputs [batch_size=3, source_len=5, d_embedding=512]  # x2 (编码器隐藏状态)
        
        # 创建位置索引序列
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        # pos_indices [1, target_len=8]
        
        # 对输入进行词嵌入和位置嵌入相加
        # self.tgt_emb(dec_inputs) [batch_size=3, target_len=8, d_embedding=512]
        # self.pos_emb(pos_indices) [1, target_len=8, d_embedding=512]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        
        # 初始化解码器隐藏状态 (对应x1 - 解码器隐藏状态):
        # dec_outputs [batch_size=3, target_len=8, d_embedding=512]
        # 这就是Seq2Seq中的x1 (解码器在各个时间步的隐藏状态)
        
        # 生成三种掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # dec_self_attn_pad_mask [batch_size=3, target_len=8, target_len=8]
        # 屏蔽<pad>位置
        
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # dec_self_attn_subsequent_mask [batch_size=3, target_len=8, target_len=8]
        # 屏蔽"未来"位置(防止看到未生成的词)
        
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask \
                                       + dec_self_attn_subsequent_mask), 0) 
        # dec_self_attn_mask [batch_size=3, target_len=8, target_len=8]
        # 组合掩码 = 填充掩码 + 后续掩码
        
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        # dec_enc_attn_mask [batch_size=3, target_len=8, source_len=5]
        # 屏蔽源序列中的<pad>位置
        
        dec_self_attns, dec_enc_attns = [], []  # 初始化
        
        # 通过解码器层 (N=6层堆叠，每层包含x1和x2的互注意力)
        for layer in self.layers:
            # 第i层处理:
            # 输入: dec_outputs [batch_size=3, target_len=8, d_embedding=512]  (x1)
            #      enc_outputs [batch_size=3, source_len=5, d_embedding=512]  (x2)
            
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, 
                                               dec_self_attn_mask, dec_enc_attn_mask)
            
            # 自注意力 (x1内部交互):
            # Q=K=V都来自dec_outputs (x1)
            # dec_self_attn [batch_size=3, n_heads=8, target_len=8, target_len=8]
            # 让目标序列中的每个词关注其他已生成的词
            
            # 互注意力 (x1关注x2):
            # Q来自dec_outputs (x1), K/V来自enc_outputs (x2)
            # dec_enc_attn [batch_size=3, n_heads=8, target_len=8, source_len=5]
            # 让解码器关注编码器中的源序列信息，实现源-目信息交互
            
            # 输出: dec_outputs [batch_size=3, target_len=8, d_embedding=512]
            
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        # 最终解码器输出 (完整的x1):
        # dec_outputs [batch_size=3, target_len=8, d_embedding=512]
        # 包含解码器的最终隐藏状态，既有自身上下文，也有编码器信息
        
        # dec_self_attns 是一个列表(6层)，每个元素的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns 是一个列表(6层)，每个元素的维度是 [batch_size, n_heads, target_len, source_len]
        
        return dec_outputs, dec_self_attns, dec_enc_attns
```

**数据流分析**：
- **x1**：dec_outputs [batch_size, target_len, d_embedding] - 解码器隐藏状态(每层更新)
- **x2**：enc_outputs [batch_size, source_len, d_embedding] - 编码器隐藏状态(固定不变)
- **关键交互**：
  - 自注意力：x1内部交互，Q=K=V来自x1
  - 互注意力：Q来自x1，K/V来自x2 (这正是Seq2Seq中的注意力机制)
  - 结果：x1在每层被x2逐步"融合"，最终输出既包含自身特征，也包含源序列信息

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
        # ============ x1 和 x2 完整数据流 ============
        # 
        # 输入:
        # enc_inputs [batch_size=3, source_len=5]  - 源序列(中文)
        # dec_inputs [batch_size=3, target_len=8]  - 目标序列(英文，包含<sos>)
        
        # 第一步: 编码器处理源序列，生成x2
        # ========================================
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        
        # enc_outputs [batch_size=3, source_len=5, d_embedding=512]
        # 这就是 x2 = torch.randn(3, 5, 512) (使用真实数据替代随机)
        # x2代表编码器对源序列的完整理解，包含5个源词的上下文表示
        
        # enc_self_attns: 列表，共6个元素(6层编码器)
        # 每个元素 [batch_size=3, n_heads=8, source_len=5, source_len=5]
        
        # 第二步: 解码器处理目标序列，并与编码器输出交互
        # ==========================================
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        
        # dec_outputs [batch_size=3, target_len=8, d_embedding=512]
        # 这就是 x1 = torch.randn(3, 8, 512) (使用真实数据替代随机)
        # x1代表解码器对目标序列的隐藏表示，包含8个目标词的上下文
        
        # dec_self_attns: 列表，共6个元素(6层解码器的自注意力)
        # 每个元素 [batch_size=3, n_heads=8, target_len=8, target_len=8]
        # x1内部的自我交互
        
        # dec_enc_attns: 列表，共6个元素(6层解码器的互注意力)
        # 每个元素 [batch_size=3, n_heads=8, target_len=8, source_len=5]
        # x1关注x2的注意力权重，通过点积计算:
        # scores = x1 @ x2^T / sqrt(d_k)  [3, 8, 8, 5]
        # weights = softmax(scores) -> 注意力分布
        # context = weights @ x2 -> x1加权融合x2
        
        # 第三步: 线性投影，输出预测概率分布
        # ===================================
        # 将x1(最后一层解码器输出)投影到词汇表大小
        # dec_outputs [batch_size=3, target_len=8, d_embedding=512]
        dec_logits = self.projection(dec_outputs)
        
        # dec_logits [batch_size=3, target_len=8, tgt_vocab_size=5000]
        # 每个目标位置都有一个概率分布，表示该位置生成各个词汇的概率
        # 例如: 位置i的概率 = softmax(dec_logits[batch, i, :])
        
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
```

**数据流总结**：

```
源序列 enc_inputs              目标序列 dec_inputs
[3, 5] token indices           [3, 8] token indices
      |                                 |
      v                                 v
   Encoder ────→ x2 ──────────→ Decoder
                 [3, 5, 512]      |
                                  v
                            互注意力融合
                            (x1 @ x2^T / sqrt(d_k))
                                  |
                                  v
                            x1 [3, 8, 512]
                                  |
                                  v
                            Projection
                                  |
                                  v
                            dec_logits
                            [3, 8, 5000]
                                  |
                                  v
                            概率分布
                            (softmax归一化)
```

**关键数据关系**：
1. **x2生成**：x2 = Encoder(enc_inputs) [batch_size, source_len=5, d_embedding]
2. **x1初始化**：x1_init = Embedding(dec_inputs) [batch_size, target_len=8, d_embedding]
3. **核心交互**：在解码器的每一层，x1都通过互注意力与x2交互：
   - `attention_scores = x1 @ x2^T / sqrt(d_k)` → [batch_size, n_heads, 8, 5]
   - `attention_weights = softmax(attention_scores)` → 概率分布
   - `context = attention_weights @ x2` → x1吸收x2的信息
4. **x1更新**：经过6层解码器，x1逐步融合x2的信息，最终包含源-目的完整理解
5. **输出**：x1投影到词表 → dec_logits → 每个位置的词汇概率分布
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

# 示例数据：使用x1, x2张量表示源和目标序列
# x1: 解码器隐藏状态 [batch_size=3, target_len=8, d_embedding=512]
# x2: 编码器隐藏状态 [batch_size=3, source_len=5, d_embedding=512]

sentences = [
    ['x2_token1 x2_token2 x2_token3', 'x1_token1 x1_token2 x1_token3 x1_token4'],
    ['x2_token2 x2_token3 x2_token4 x2_token5', 'x1_token1 x1_token2 x1_token3'],
    ['x2_token1 x2_token2', 'x1_token1 x1_token2 x1_token3 x1_token4 x1_token5'],
    ['x2_token3 x2_token4 x2_token5', 'x1_token1 x1_token2'],
    ['x2_token1 x2_token3 x2_token5', 'x1_token1 x1_token2 x1_token3 x1_token4']
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
# 测试：生成一个大小为1的批次
# enc_inputs [batch_size=1, source_len=5]  - x2对应的源序列token索引
# dec_inputs [batch_size=1, target_len=8]  - x1对应的目标序列token索引(包含<sos>)
enc_inputs, dec_inputs, target_batch = corpus.make_batch(batch_size=1, test_batch=True) 

print(f"enc_inputs (x2源) shape: {enc_inputs.shape}")  # [1, 5]
print(f"dec_inputs (x1目标) shape: {dec_inputs.shape}")  # [1, 8]
print(f"enc_inputs (x2): {enc_inputs}")  # 源序列索引
print(f"dec_inputs (x1): {dec_inputs}")  # 目标序列索引

# 进行推理：通过编码器生成x2，通过解码器生成x1
predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

print(f"\n推理结果:")
print(f"Encoder输出 (x2) shape: [1, 5, 512]")
print(f"Decoder输出 (x1) shape: [1, 8, 512]")
print(f"Projection输出 (dec_logits) shape: {predict.shape}")  # [1, 8, vocab_size]

# 将预测结果维度重塑为[seq_len * batch_size, vocab_size]
predict = predict.view(-1, len(corpus.tgt_vocab))

# 找到每个位置概率最大的单词的索引
predict = predict.data.max(1, keepdim=True)[1]

print(f"\n预测的token索引: {predict.squeeze().tolist()}")
print(f"预测的词汇: {[corpus.tgt_idx2word.get(idx, '<unk>') for idx in predict.squeeze()]}")

# 解码输入的源序列(x2源)
input_tokens = [corpus.src_idx2word.get(idx, '<unk>') for idx in enc_inputs[0]]
print(f"\n输入的源序列(x2): {' '.join(input_tokens)}")

# 解码预测的目标序列(x1目标)
translated_tokens = [corpus.tgt_idx2word.get(idx.item(), '<unk>') for idx in predict.squeeze()]
print(f"预测的目标序列(x1): {' '.join(translated_tokens)}")

print(f"\n注意力权重分解:")
print(f"编码器自注意力权重数: {len(enc_self_attns)}层 (6层编码器)")
print(f"解码器自注意力权重数: {len(dec_self_attns)}层 (6层解码器)")
print(f"解码器-编码器互注意力权重数: {len(dec_enc_attns)}层 (x1关注x2的权重)")
print(f"x1关注x2的权重shape: {dec_enc_attns[0].shape}")  # [1, n_heads, target_len, source_len]
```

**关键概念总结**：
- **x2**: 编码器输出 [batch_size, source_len, d_embedding] - 源序列的完整上下文表示
- **x1**: 解码器输出 [batch_size, target_len, d_embedding] - 目标序列与源序列交互后的表示
- **核心交互**：解码器中的互注意力层让x1通过 `attention = softmax(x1 @ x2^T / sqrt(d_k)) @ x2` 来融合源序列信息
- **自回归生成**：要实现完整翻译，需要循环生成：每次将上一步预测的token作为下一步的x1输入，直到生成<eos>或达到最大长度

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

## 完整总结与对比

### 重要概念回顾

| 概念 | 说明 |
|-----|-----|
| 词嵌入 | 将词转为向量表示 |
| 位置编码 | 为词向量加入位置信息 |
| 自注意力 | 序列内部位置互相关注 |
| 交互注意力 | 解码器关注编码器输出 |
| 多头自注意力 | 多个独立注意力头并行关注 |
| 残差连接 | 直接传递原始信息（解决梯度消失） |
| 层归一化 | 稳定训练，加速收敛 |
| 前馈网络 | 非线性特征变换 |
| 掩码 | 屏蔽某些位置的注意力 |
| 教师强制 | 训练时用真实目标序列作为输入 |

### ChatGPT 所用技术

**ChatGPT 使用的是自注意力机制（Self-Attention）**

- 基于Transformer架构
- 核心是多层自注意力和前馈网络
- 不使用传统的RNN编码器-解码器结构
- 无需交互注意力

---

## 完整工作流

1. **数据准备**：词汇表构建、句子编码、填充、批处理
2. **模型初始化**：创建 Transformer 实例
3. **训练**：前向传播 → 计算损失 → 反向传播 → 参数更新
4. **推理**：编码源序列 → 解码生成目标序列
5. **评估**：对比翻译结果和目标翻译

---

## 学习路径建议

### 从基础到进阶的学习顺序

1. **理论基础**（本文档第1-8部分）
   - 理解张量、序列表示
   - 掌握注意力机制的本质（Query、Key、Value）
   - 区分交互注意力和自注意力
   - 理解多头自注意力的优势

2. **组件实现**（本文档第9-15部分）
   - 实现基础的ScaledDotProductAttention
   - 实现MultiHeadAttention
   - 实现前馈网络和位置编码
   - 实现掩码机制

3. **模型构建**
   - 实现EncoderLayer和DecoderLayer
   - 堆叠完整的Encoder和Decoder
   - 组合成完整的Transformer模型

4. **应用实践**
   - 准备数据和语料库
   - 训练模型
   - 进行推理和评估

### 核心要点总结

**1. 为什么需要Attention？**
- 传统RNN/LSTM受限于顺序处理，难以捕捉长距离依赖
- 注意力机制允许模型直接关注序列中的任意位置
- 大大提高了模型对长距离依赖的建模能力

**2. 为什么需要多头注意力？**
- 单头注意力只能从一个角度关注序列
- 多头注意力能从多个子空间同时关注，增加特征的多样性
- 不同的头可以学习不同的关注模式和特征

**3. 为什么需要位置编码？**
- 自注意力是全局并行的，无法保留位置顺序信息
- 位置编码为每个位置注入唯一的标识
- 模型能够感知序列中的相对位置关系

**4. 为什么需要掩码？**
- 填充掩码：避免模型关注无意义的填充符号
- 后续掩码：保证解码器的因果性，防止"作弊"看到未来信息

**5. 为什么Transformer优于Seq2Seq？**
- 并行计算：所有位置同时处理，而RNN必须顺序处理
- 长距离依赖：自注意力天然支持全局依赖
- 可扩展性：更容易扩展到大规模模型（如ChatGPT）

---

## 架构图总览

### Transformer 完整流程图

```
源序列输入                      目标序列输入
    |                               |
    v                               v
+--------+                      +--------+
| 词嵌入 |                      | 词嵌入 |
+--------+                      +--------+
    |                               |
    v                               v
+--+--+                         +--+--+
|位置编码|                         |位置编码|
+-------+                         +-------+
    |                               |
    v                               v
+----------+                     +----------+
| 编码器    | (N层)              | 解码器    | (N层)
| Encoder  |--------+     +----> | Decoder  |
+----------+        |     |      +----------+
                    |     |
                    +-----+
                    互注意力
                    
                    |
                    v
                  +--------+
                  | 线性层 |
                  +--------+
                    |
                    v
                 +--------+
                 | Softmax|
                 +--------+
                    |
                    v
                 输出分布
```

### 每层子结构

```
编码器层 (EncoderLayer)          解码器层 (DecoderLayer)
┌─────────────────────┐         ┌──────────────────────┐
│ 多头自注意力         │         │ 多头自注意力(Masked)  │
│ (Self-Attention)    │         │ (Self-Attention)     │
└──────────┬──────────┘         └──────────┬───────────┘
           │                               │
           v                               v
    ┌─────────────┐                 ┌─────────────┐
    │ + & LayerNorm│                 │ + & LayerNorm│
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           v                               v
┌─────────────────────┐         ┌──────────────────────┐
│ 前馈网络             │         │ 编码器-解码器注意力   │
│ (FeedForward Net)   │         │ (Encoder-Decoder Attn)
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           v                               v
    ┌─────────────┐                 ┌─────────────┐
    │ + & LayerNorm│                 │ + & LayerNorm│
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           v                               v
        输出                            ┌──────────────────┐
                                        │ 前馈网络          │
                                        │ (FeedForward Net) │
                                        └────────┬─────────┘
                                                 │
                                                 v
                                          ┌─────────────┐
                                          │ + & LayerNorm│
                                          └──────┬──────┘
                                                 │
                                                 v
                                              输出
```

---

这样就完成了从基础概念、理论理解、组件实现、模型构建到训练推理的完整 Transformer 架构学习和实践！
