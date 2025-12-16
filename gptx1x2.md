# GPT模型实现完整指南

## 目录
1. [为什么实现GPT](#为什么实现gpt)
2. [GPT关键组件](#gpt关键组件)
3. [完整代码实现](#完整代码实现)
4. [语料库处理](#语料库处理)
5. [模型训练](#模型训练)
6. [文本生成](#文本生成)
7. [实现总结](#实现总结)

---

## 为什么实现GPT

### Transformer vs GPT

之前已经实现了完整的Transformer（编码器+解码器），现在为什么还要单独实现GPT呢？

**关键区别：**

| 特性 | Transformer | GPT |
|-----|-----------|-----|
| 结构 | Encoder（处理x2） + Decoder（处理x1） | 仅Decoder（处理x1） |
| 用途 | 机器翻译、文本摘要（Seq2Seq，x2→x1） | 文本生成、语言建模（x1自回归） |
| 输入 | x1（目标序列）+ x2（源序列） | 仅x1（历史token的自回归） |
| 数据流 | x1和x2交互（cross-attention，x1←x2） | 仅x1自注意力（masked） |
| 信息交互 | 源序列x2→目标序列x1 | x1序列内部（前→后） |

### 为什么单独实现

1. **结构差异**：GPT只需解码器部分，Transformer包含编码器和跨源-目标的互注意力。
2. **代码简化**：GPT去掉编码器和cross-attention，结构更精简，便于理解和复用。
3. **用途不同**：
   - Transformer：两个序列的对齐/转换任务（输入→输出）
   - GPT：单向自回归生成（历史→未来）
4. **工程实践**：GPT是当今NLP主流（ChatGPT、GPT-4等），单独实现更符合实际应用。

**总结**：虽然底层组件类似，但GPT的数据流、掩码机制和使用方式与完整Transformer完全不同，所以需要单独实现一次。

---

## GPT关键组件

GPT模型需要以下7个核心组件：

### 组件1：多头自注意力
通过`ScaledDotProductAttention`实现缩放点积注意力机制，然后通过`MultiHeadAttention`实现多头自注意力机制。

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()        
    
    def forward(self, Q, K, V, attn_mask):
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

### 组件2：多头自注意力机制

```python
d_embedding = 512  # Embedding的维度
n_heads = 8        # Multi-Head Attention中头的个数
d_k = 64           # K维度
d_v = 64           # V维度

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_k * n_heads)
        self.W_K = nn.Linear(d_embedding, d_k * n_heads)
        self.W_V = nn.Linear(d_embedding, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    
    def forward(self, Q, K, V, attn_mask): 
        residual, batch_size = Q, Q.size(0)
        
        # 线性变换和重塑
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        
        # 将注意力掩码复制到多头
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        
        # 缩放点积注意力
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        
        # 多头拼接
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) 
        
        # 输出线性投影
        output = self.linear(context) 
        
        # 残差连接 + LayerNorm
        output = self.layer_norm(output + residual)
        
        return output, weights
```

### 组件3：逐位置前馈网络

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)
    
    def forward(self, inputs): 
        residual = inputs  
        
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) 
        output = self.conv2(output).transpose(1, 2) 
        
        output = self.layer_norm(output + residual) 
        
        return output
```

### 组件4：正弦位置编码表

```python
def get_sin_code_table(n_position, embedding_dim):
    # n_position: 输入序列的最大长度
    # embedding_dim: 词嵌入向量的维度
    
    sinusoid_table = np.zeros((n_position, embedding_dim))    
    
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle    
    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])   # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])   # dim 2i+1 奇数维
    
    return torch.FloatTensor(sinusoid_table)
```

**公式**：
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

### 组件5：填充掩码

```python
def get_attn_pad_mask(seq_q, seq_k):
    # seq_q 和 seq_k 的维度是 [batch_size, seq_len]
    # <PAD> token的编码值为0
    
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    
    return pad_attn_mask
```

**作用**：屏蔽填充位置，防止模型关注无效信息。

### 组件6：后续掩码

```python
def get_attn_subsequent_mask(seq):
    # seq 的维度是 [batch_size, seq_len]
    # 生成上三角矩阵：当前位置之后的都是1（需要掩码）
    
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    return torch.from_numpy(subsequent_mask).bool()
```

**作用**：防止解码器在自注意力中"偷窥"未来信息（当前位置后的所有位置都被掩码）。这是GPT等自回归模型的关键。

### 组件7：解码器层

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()        
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self, dec_inputs, attn_mask):
        # dec_inputs: [batch_size, seq_len, d_embedding] - x1 隐藏状态
        # attn_mask: [batch_size, seq_len, seq_len] - 后续掩码（防止看未来）
        
        # 仅 masked self-attention，无 cross-attention（这是 GPT 的特点）
        # x1 仅关注自己和之前的 x1_token，实现自回归
        dec_outputs, dec_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                    dec_inputs, attn_mask)
        # dec_outputs: [batch_size, seq_len, d_embedding] - x1 经过多头注意力融合
        
        # 前馈网络进一步处理 x1
        dec_outputs = self.pos_ffn(dec_outputs)
        # dec_outputs: [batch_size, seq_len, d_embedding] - 最终的 x1 输出
        
        return dec_outputs, dec_attn
```

**注意**：GPT的解码器层只有masked self-attention + FFN，**没有cross-attention**。

---

## 完整代码实现

### 解码器类

```python
n_layers = 6  # 设置Decoder的层数

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(Decoder, self).__init__()
        # 词嵌入层
        self.src_emb = nn.Embedding(vocab_size, d_embedding)  
        # 位置编码层（可选：用Embedding或get_sin_code_table）
        self.pos_emb = nn.Embedding(max_seq_len, d_embedding)
        # N层解码器       
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]) 
    
    def forward(self, dec_inputs):
        # dec_inputs: [batch_size, seq_len] - x1_token 索引序列
        
        # 创建位置索引，支持batch
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device).unsqueeze(0).expand(dec_inputs.size(0), -1)
        # positions: [batch_size, seq_len] - 位置编码用于捕捉序列顺序
        
        # x1 词嵌入 + 位置编码
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions)
        # inputs_embedding: [batch_size, seq_len, d_embedding=512] - x1 初始隐藏状态
        
        # 生成后续掩码（防止看未来）
        # 这是 GPT 自回归性的关键：每个位置只能关注自己和之前的位置
        attn_mask = get_attn_subsequent_mask(dec_inputs).to(dec_inputs.device)
        # attn_mask: [batch_size, seq_len, seq_len]
        
        # x1 经过多层解码器更新
        dec_outputs = inputs_embedding
        
        # 通过N层解码器（每层都是 masked self-attention + FFN）
        for layer in self.layers:
            # x1 在每层自我更新（仅关注自己和历史 x1_token）
            dec_outputs, dec_attn = layer(dec_outputs, attn_mask) 
        
        return dec_outputs  # [batch_size, seq_len, d_embedding=512] - 最终的 x1 隐藏状态
```

### GPT模型

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(GPT, self).__init__()
        self.decoder = Decoder(vocab_size, max_seq_len)  # 解码器
        self.projection = nn.Linear(d_embedding, vocab_size)  # 输出层
    
    def forward(self, dec_inputs):
        # dec_inputs: [batch_size, seq_len] - x1_token 索引序列
        
        # 通过解码器处理 x1_token 序列
        dec_outputs = self.decoder(dec_inputs)  # [batch_size, seq_len, d_embedding=512] - x1 隐藏状态
        
        # 将 x1 隐藏状态投影到词表
        logits = self.projection(dec_outputs)   # [batch_size, seq_len, vocab_size] - 每个位置的 x1_token 概率分布
        
        return logits
```

**说明**：
- 输入：token索引，形状[batch_size, seq_len]
- 输出：logits，形状[batch_size, seq_len, vocab_size]，可用softmax转为概率分布

---

## 语料库处理

### LanguageCorpus类

```python
from collections import Counter

class LanguageCorpus:
    def __init__(self, sentences):
        # sentences: 句子列表，每句由 x1_token 或 x2_token 组成
        # 例如：['x1_token1 x1_token2 x1_token3', 'x1_token2 x1_token3 x1_token4']
        self.sentences = sentences
        # 计算最大句子长度，加2容纳<sos>和<eos>
        # 最终序列形状为 [batch_size, seq_len]
        self.seq_len = max([len(sentence.split()) for sentence in sentences]) + 2
        self.vocab = self.create_vocabulary()
        self.idx2word = {v: k for k, v in self.vocab.items()}
    
    def create_vocabulary(self):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        counter = Counter()
        # 统计单词频率
        for sentence in self.sentences:
            words = sentence.split()
            counter.update(words)
        # 为每个单词分配唯一索引
        for word in counter:
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab
    
    def make_batch(self, batch_size, test_batch=False):
        input_batch, output_batch = [], []
        # 随机选择句子
        sentence_indices = torch.randperm(len(self.sentences))[:batch_size]
        
        for index in sentence_indices:
            sentence = self.sentences[index]
            # 转换为索引序列：<sos> + x1_tokens + <eos> + padding
            # 表示解码器隐藏状态 x1 的 token 序列
            seq = [self.vocab['<sos>']] + [self.vocab[word] for word in sentence.split()] + [self.vocab['<eos>']]
            seq += [self.vocab['<pad>']] * (self.seq_len - len(seq))
            
            # 输入为seq[:-1]（去掉最后一个token），形状 [batch_size, seq_len]
            # 输出为seq[1:]（去掉第一个token），形状 [batch_size, seq_len]
            # 实现"前n-1个x1_token预测第n个x1_token"的自回归训练
            input_batch.append(seq[:-1])
            output_batch.append(seq[1:])
        
        return torch.LongTensor(input_batch), torch.LongTensor(output_batch)
```

**说明**：
- 输入batch：[batch_size, seq_len]，用于预测
- 输出batch：[batch_size, seq_len]，目标标签
- 本质：每句话向右移一位作为标签，实现"前n-1词预测第n词"

### 读取语料库

```python
with open("lang.txt", "r") as file:  # 从文件读取语料（每行为一句包含 x1_token 的文本）
    sentences = [line.strip() for line in file.readlines()]
    # 例如：['x1_token1 x1_token2 x1_token3', 'x1_token2 x1_token3 x1_token4 x1_token5']

corpus = LanguageCorpus(sentences)  # 创建语料库
vocab_size = len(corpus.vocab)      # 词汇表大小（包含<pad>, <sos>, <eos>和各个x1_token）
max_seq_len = corpus.seq_len        # 最大句子长度（用于位置编码和序列补全）

print(f"语料库词汇表大小: {vocab_size}")
print(f"最长句子长度 (含<sos>和<eos>): {max_seq_len}")

---

## 模型训练

```python
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(vocab_size, max_seq_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 500
batch_size = 32

for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零
    
    # 创建训练数据
    # inputs: [batch_size, seq_len] - x1_token索引序列（前n-1位置）
    # targets: [batch_size, seq_len] - x1_token索引序列（后n-1位置，用于监督）
    inputs, targets = corpus.make_batch(batch_size)
    inputs, targets = inputs.to(device), targets.to(device)
    
    # 前向传播
    outputs = model(inputs)  # [batch_size, seq_len, vocab_size] - 预测每个位置的x1_token概率
    
    # 计算损失（展平为[batch_size*seq_len, vocab_size]和[batch_size*seq_len]）
    # 目标：最小化预测 x1_token 与真实 x1_token 的交叉熵
    loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
    
    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1:04d} cost = {loss:.6f}")
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
```

**训练流程说明**：
1. 每轮随机采样一个batch的句子。
2. 前向传播获取logits。
3. 计算交叉熵损失（比较预测和目标）。
4. 反向传播更新参数。
5. 定期打印损失以监控训练进度。

---

## 文本生成

### 自回归生成函数

```python
def generate_text(model, input_str, max_len=50):
    model.eval()  # 评估模式
    
    # 将输入 x1_token 转换为索引
    # input_str: 起始 x1_token 列表，如 ['x1_token1']
    input_tokens = [corpus.vocab[token] for token in input_str]
    output_tokens = input_tokens.copy()
    
    with torch.no_grad():  # 禁用梯度计算
        for _ in range(max_len):
            # 将当前已生成的 x1_token 索引序列转为张量
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            # inputs: [1, current_len] - 当前积累的 x1_token 序列
            
            # 前向传播，预测下一个 x1_token
            outputs = model(inputs)  # [1, current_len, vocab_size]
            
            # 取最后一个位置的最大概率 x1_token（贪心采样）
            _, next_token = torch.max(outputs[:, -1, :], dim=-1)
            next_token = next_token.item()
            
            # 如果生成<eos>则停止
            if next_token == corpus.vocab["<eos>"]:
                break
            
            # 将新生成的 x1_token 添加到输出序列
            output_tokens.append(next_token)
    
    # 将 x1_token 索引序列转回文本
    output_str = " ".join([corpus.idx2word[token] for token in output_tokens])
    return output_str

# 使用示例
# 从起始 x1_token 开始自回归生成后续 x1_token
input_str = ["x1_token1"]
generated_text = generate_text(model, input_str)
print("生成的 x1 序列：", generated_text)
```

**生成过程说明**：
1. 从输入token开始（如["Python"]）。
2. 每次预测下一个token，并加到已生成序列末尾。
3. 重复直到生成<eos>或到达max_len。
4. 输出完整的生成文本。

---

## 实现总结

### 核心要点

1. **GPT的本质**：仅用Transformer解码器，实现自回归语言建模。
2. **关键区别**：
   - Transformer（Encoder+Decoder）：源→目标序列转换
   - GPT（仅Decoder）：历史→未来token预测

3. **掩码机制**：后续掩码防止模型"看未来"，是自回归的基础。

4. **数据流**：
   - 输入：历史token索引
   - 嵌入+位置编码
   - N层masked self-attention + FFN
   - Projection到词表
   - 输出：下一个token的概率分布

5. **训练方式**："前n-1词预测第n词"，逐步优化模型参数。

6. **推理方式**：自回归采样，支持任意起始词生成连贯文本。

### 完整流程

```
文本语料库 (lang.txt，包含x1_token)
    ↓
x1_token索引化 + LanguageCorpus
    ↓
x1批处理 [batch_size, seq_len]
    ↓
GPT模型 (Decoder处理x1 + Projection输出概率)
    ↓
训练循环：预测x1 → 损失 → 反向传播 → 优化
    ↓
训练好的模型（学会x1自回归）
    ↓
generate_text (自回归采样生成x1)
    ↓
生成的x1_token序列
```

### 实现步骤总结

1. **数据准备**：读取包含x1_token的文本文件，构建词表和批处理工具。
2. **组件实现**：实现7个关键组件（多头注意力、前馈、位置编码、掩码、解码器层）。
3. **模型搭建**：组合Decoder（处理x1隐藏状态）和Projection层（输出x1_token概率）形成完整GPT。
4. **模型训练**：编写训练循环，用"前n-1个x1_token预测第n个x1_token"方式持续优化参数。
5. **文本生成**：实现自回归生成函数，从起始x1_token自回归生成后续x1_token。

---

**一句话总结**：GPT是"用Transformer解码器做x1隐藏状态的自回归语言建模"，核心是x1_token数据流、后续掩码防止"看未来"、多层解码器自注意力堆叠、以及自回归生成。
