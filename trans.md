# Seq2Seq、注意力机制与Transformer学习总结

## 一、基础概念

### 1. x1 和 x2 张量
- x1 = torch.randn(2, 3, 4)：形状(batch_size, seq_len1, feature_dim)
- x2 = torch.randn(2, 5, 4)：形状(batch_size, seq_len2, feature_dim)

**生活中的例子：**
- x1：每个学生有3门课程，每门课程有4个维度的特征（分数、作业分、出勤率、参与度）
- x2：每个学生有5个兴趣班，每个兴趣班也有4个维度的特征（表现、兴趣度、老师评分、完成度）

### 2. Seq2Seq架构中的 x1 和 x2

在Seq2Seq架构中，点积注意力通常用于将编码器的隐藏状态与解码器的隐藏状态联系起来：
- **x1**：解码器在各个时间步的隐藏状态，形状为(batch_size, seq_len1, feature_dim)
- **x2**：编码器在各个时间步的隐藏状态，形状为(batch_size, seq_len2, feature_dim)

## 二、注意力机制基础

### 1. Attention类实现
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

### 2. DecoderWithAttention 类

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

### 3. Seq2Seq模型

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

## 三、Query、Key、Value 概念

### 1. 三个关键部分

- **查询（Query）**：当前需要处理的信息。模型根据查询向量在输入序列中查找相关信息。
- **键（Key）**：来自输入序列的一组表示。用于根据查询向量计算注意力权重。
- **值（Value）**：来自输入序列的一组表示。用于根据注意力权重计算加权和。

### 2. Seq2Seq中的 Q、K、V 映射

- x1（解码器隐藏状态）→ Query（q）
- x2（编码器隐藏状态）→ Key（k）和 Value（v）

在基础点积注意力中，k 和 v 通常使用同一个张量（x2）。

### 3. 注意力计算过程

1. 用查询向量与各个键向量计算相似性（点积），得到注意力分数
2. 对分数做softmax归一化，得到权重
3. 用权重对所有值向量加权求和，得到输出向量
4. 输出向量蕴含输入序列中与查询最相关的信息

## 四、交互注意力 vs 自注意力

### 1. 交互注意力（Encoder-Decoder Attention）

**特点：**
- Query 来自解码器（x1）
- Key/Value 来自编码器（x2）
- 解码器每一步都关注编码器输出的所有部分

**应用：**
- 传统Seq2Seq模型
- 机器翻译、文本摘要等任务

**类比：**
主课（x1，解码器）在评估学生能力时，会"关注"兴趣班（x2，编码器）的不同表现，通过点积注意力机制，判断哪些兴趣班的经历对主课成绩影响最大，从而加权融合这些信息，帮助更好地评估和生成主课成绩。

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

## 五、多头自注意力（Multi-Head Self-Attention）

### 1. 计算过程

1. **初始化**：设定多个头，每个头有独立的线性变换权重
2. **线性变换**：对Q、K、V进行多次线性变换，每次使用不同的权重矩阵
3. **缩放点积注意力**：每组Q、K、V独立计算注意力，每个头生成一个加权输出
4. **合并**：将所有头的输出拼接，再进行一次线性变换

### 2. 优势

- 通过同时学习多个子空间的特征，提高模型捕捉长距离依赖的能力
- 能在不同的语义层次上并行关注信息
- 增强模型表达能力

### 3. 类比说明

在自注意力机制下（如Transformer）：
- 不再区分"课程"和"兴趣班"，统一作为"序列的位置"
- 多头自注意力 = 多个老师用不同标准分析同一组课程/兴趣班的表现
- 每个头都能从不同角度融合序列内所有位置的信息
- 最后综合所有老师的意见，得到最丰富的学生能力描述

### 4. 关键概念

- x1、x2、...、xn 代表同一个序列的不同位置
- 不区分学生（batch中的每个样本独立处理）
- 每个头用自己的线性变换关注序列的不同特征组合

## 六、ChatGPT 所用技术

**ChatGPT 使用的是自注意力机制（Self-Attention）**

- 基于Transformer架构
- 核心是多层自注意力和前馈网络
- 不使用传统的RNN编码器-解码器结构
- 无需交互注意力

## 七、总结

| 概念 | 说明 |
|-----|-----|
| **x1（解码器隐藏状态）** | 代表需要输出的内容，对应Query |
| **x2（编码器隐藏状态）** | 代表输入的背景信息，对应Key/Value |
| **Query** | 当前需要处理/关注的内容 |
| **Key** | 用于计算相似度的基准 |
| **Value** | 实际被加权融合的信息 |
| **交互注意力** | 解码器关注编码器，用于Seq2Seq |
| **自注意力** | 序列自身位置互相关注，用于Transformer |
| **多头自注意力** | 多个头从不同角度学习，增强表达能力 |

## 八、位置编码（Positional Encoding）

在Transformer中，位置编码的目的是为模型提供序列中词的位置信息。由于自注意力机制是全局的，无法天然理解词的顺序，因此需要显式加入位置信息。

位置编码通常使用正弦和余弦函数，并与词嵌入向量相加，补充序列顺序信息。

## 九、注意力掩码（Attention Mask）

### 1. 填充注意力掩码（Padding Attention Mask）
- 作用：屏蔽掉输入序列中填充（padding）的位置，防止模型关注无效信息。
- 应用场景：Transformer编码器、解码器的所有注意力层。

### 2. 前瞻/后续注意力掩码（Look-ahead/Subsequent Attention Mask）
- 作用：在自回归生成时，防止模型看到未来的信息，保证生成的因果性。
- 应用场景：Transformer解码器的自注意力层。

总结：  
- 编码器只需填充掩码。
- 解码器既需填充掩码，也需前瞻掩码。

## 十、Transformer编码器内部结构

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

### FFN内部结构
```python
def FeedForwardNetwork(x):
    y = Linear1(x)         # 升维（如512→2048）
    y = ReLU(y)            # 非线性激活
    y = Linear2(y)         # 降维（如2048→512）
    return y
```

- 激活函数作用：引入非线性，使网络能拟合更复杂的函数关系。
- ReLU：简单高效，当x>0时输出x，否则输出0。

## 十一、Transformer解码器内部结构

### 解码器单层结构
1. 多头自注意力（Masked Multi-Head Self-Attention）
   - 只能关注已生成的目标序列（用掩码防止"看未来"）
2. 编码器-解码器注意力（Encoder-Decoder Attention）
   - Query来自解码器自注意力输出，Key/Value来自编码器输出
3. 前馈神经网络（FFN）
4. 每个子层都有残差连接+层归一化

### 解码器输入与目标序列的关系

训练时：
- decoder_input: <sos> KaGe likes XiaoBing  
- Target:        KaGe likes XiaoBing <eos>

作用：模型每一步都用当前输入预测"下一个词"，最终生成完整目标序列。

### 训练与推理的区别
- 训练（教师强制）：用真实目标序列作为解码器输入
- 推理（自回归生成）：用模型自己已生成的词作为当前输入

## 十二、Transformer输出层

1. 线性层：将解码器最后输出映射到词汇表大小的空间，得到每个词的分数。
2. softmax层：将分数转为概率分布（所有概率和为1）。
3. 最后得到每个位置上所有词的概率分布，可用于生成或分类。

这个过程与skip-gram模型预测目标词w2的过程本质相同（特征向量→线性变换→softmax）。

## 十三、RNN、LSTM、GRU简述

三者都是处理序列数据的循环神经网络：

- RNN（Recurrent Neural Network）：基础版本，每步输出依赖当前输入和上一步隐藏状态。容易梯度消失/爆炸，难以捕捉长距离依赖。

- LSTM（Long Short-Term Memory）：引入"门控机制"和"细胞状态"，能有效记忆和遗忘信息，解决了RNN的长距离依赖问题。

- GRU（Gated Recurrent Unit）：结构比LSTM更简单，只有重置门和更新门，也能捕捉长距离依赖，计算更高效。

本质：通过"循环"结构把前面信息传递到后面，实现序列建模。

## 十四、Skip-gram、Seq2Seq、Transformer对比

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

## 十五、重要概念回顾

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

---

注：本文档基于对Seq2Seq、注意力机制和Transformer的深入学习与讨论总结而成。
