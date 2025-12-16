# WikiText2 GPT训练完整指南

## 目录
1. [数据准备与词表构建](#数据准备与词表构建)
2. [自定义数据集](#自定义数据集)
3. [批处理与DataLoader](#批处理与dataloader)
4. [模型训练](#模型训练)
5. [验证与模型保存](#验证与模型保存)
6. [文本生成（Beam Search）](#文本生成beam-search)
7. [总结](#总结)

---

## 数据准备与词表构建

### 加载WikiText2数据集

```python
from torchtext.datasets import WikiText2  # 导入WikiText2
from torchtext.data.utils import get_tokenizer  # 导入Tokenizer分词工具
from torchtext.vocab import build_vocab_from_iterator  # 导入Vocabulary工具
from torch.utils.data import DataLoader, Dataset  # 导入Pytorch的DataLoader和Dataset

tokenizer = get_tokenizer("basic_english")  # 定义数据预处理所需的Tokenizer
train_iter = WikiText2(split='train')  # 加载WikiText2数据集的训练部分

# 定义一个生成器函数，用于将数据集中的文本转换为tokens
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

# 创建词汇表，包括特殊tokens: "<pad>", "<sos>", "<eos>"
vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                  specials=["<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<pad>"])

# 打印词汇表信息
print("词汇表大小：", len(vocab))
print("词汇示例(word to index): ", {word: vocab[word] for word in ["<pad>", "<sos>", "<eos>", "the", "apple"]})
```

**说明：**
- `WikiText2`：加载WikiText2语料（英文大规模文本）。
- `get_tokenizer`：分词器（这里用basic_english）。
- `build_vocab_from_iterator`：根据分词结果构建词表。
- 词表包含特殊符号：`<pad>`（填充）、`<sos>`（句首）、`<eos>`（句尾）。
- 未知词默认映射为`<pad>`。

---

## 自定义数据集

### WikiDataset类

```python
from torch.utils.data import Dataset  # 导入Dataset类
max_seq_len = 256  # 设置序列的最大长度

# 定义一个处理WikiText2数据集的自定义数据集类
class WikiDataset(Dataset):
    def __init__(self, data_iter, vocab, max_len=max_seq_len):
        self.data = []        
        for sentence in data_iter:  # 遍历数据集，将文本转换为tokens
            # 对每个句子进行Tokenization，截取长度为max_len-2，为<sos>和<eos>留出空间
            tokens = tokenizer(sentence)[:max_len - 2]
            tokens = [vocab["<sos>"]] + vocab(tokens) + [vocab["<eos>"]]  # 添加<sos>和<eos>
            self.data.append(tokens)  # 将处理好的tokens添加到数据集中
    
    def __len__(self):  # 定义数据集的长度
        return len(self.data)    
    
    def __getitem__(self, idx):  # 定义数据集的索引方法（即抽取数据条目）        
        source = self.data[idx][:-1]  # 获取当前数据，并将<eos>移除，作为源(source)数据
        target = self.data[idx][1:]   # 获取当前数据，并将<sos>移除，作为目标(target)数据（右移1位）       
        return torch.tensor(source), torch.tensor(target)  # 转换为tensor并返回

train_dataset = WikiDataset(train_iter, vocab)  # 创建训练数据集
print(f"Dataset数据条目数: {len(train_dataset)}")
sample_source, sample_target = train_dataset[100]
print(f"输入序列张量样例: {sample_source}")
print(f"目标序列张量样例: {sample_target}")
decoded_source = ' '.join(vocab.lookup_tokens(sample_source.tolist()))
decoded_target = ' '.join(vocab.lookup_tokens(sample_target.tolist()))
print(f"输入序列样例文本: {decoded_source}")
print(f"目标序列样例文本: {decoded_target}")
```

**输出示例：**
```
Dataset数据条目数: 36718
输入序列张量样例: tensor([    1,  2659,  3478, 17569,  9098])
目标序列张量样例: tensor([ 2659,  3478, 17569,  9098,     2])
输入序列样例文本: <sos> 96 ammunition packing boxes
目标序列样例文本: 96 ammunition packing boxes <eos>
```

**说明：**
- `source`：去掉最后一个token（即<eos>），作为输入
- `target`：去掉第一个token（即<sos>），作为目标（右移一位）
- 这正是自回归语言建模的标准格式："前n-1个token预测第n个token"

---

## Dataset vs DataLoader

| 特性 | Dataset | DataLoader |
|-----|---------|------------|
| 作用 | 数据的组织与索引 | 批量采样、打乱、并行加载 |
| 实现 | 继承Dataset，实现`__len__`和`__getitem__` | 包装Dataset，自动批量化 |
| 职责 | 单条数据的读取和预处理 | 批量、打乱、pad、多线程 |
| 关系 | 数据的"内容和结构" | "批量和调度"工具 |

**一句话理解：**
- **Dataset** = 数据的"单条读取规则"
- **DataLoader** = "批量采样+打乱+补齐+多线程加载"工具

---

## 批处理与DataLoader

### pad_sequence函数

```python
# 定义pad_sequence函数，用于将一批序列补齐到相同长度
def pad_sequence(sequences, padding_value=0, length=None):
    # 计算最大序列长度，如果length参数未提供，则使用输入序列中的最大长度
    max_length = max(len(seq) for seq in sequences) if length is None else length
    # 创建一个具有适当形状的全零张量，用于存储补齐后的序列
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)    
    # 遍历序列，将每个序列的内容复制到张量result中
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    return result
```

### collate_fn函数

```python
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)    
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=vocab["<pad>"], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab["<pad>"], length=max_length)    
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建一个训练数据加载器，使用自定义的collate_fn函数
batch_size = 3
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
```

**说明：**
- `pad_sequence`负责单批次补齐
- `collate_fn`负责拆分、批量pad和组装
- 这样可以高效处理变长文本，适配GPT等模型训练

---

## 模型训练

### 训练主循环

```python
import torch.optim as optim  # 导入优化器

device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置设备
model = GPT(len(vocab), max_seq_len).to(device)  # 创建GPT模型实例（使用之前定义的GPT类）
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])  # 损失函数，忽略<pad>
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 优化器
epochs = 2  # 训练轮次

for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (source, target) in enumerate(train_dataloader):  # 用dataloader加载数据
        inputs, targets = source.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(inputs)  # 获取模型输出
        loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        epoch_loss += loss.item()  # 积累每轮损失      
        if (batch_idx + 1) % 1000 == 0:  # 每1000个批次打印一次损失
            print(f"Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item()}")    
    epoch_loss /= len(train_dataloader)  # 每轮打印一次损失
    print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss}")
```

**训练日志示例：**
```
Batch 1000/12240, Loss: 7.157247543334961
Batch 2000/12240, Loss: 3.339968204498291
Batch 3000/12240, Loss: 5.498887538909912
Batch 4000/12240, Loss: 6.358556747436523
Batch 5000/12240, Loss: 2.53767728805542
```

**说明：**
- 损失有高有低，属于大语料和大词表下的正常现象（尤其是训练初期）
- 损失整体有下降趋势，说明模型在学习
- 训练用的是之前定义的GPT模型类（Decoder+Projection），不是临时新建的

---

## 验证与模型保存

### 验证集处理

```python
# 加载验证集
valid_iter = WikiText2(split='valid')  # 加载WikiText2数据集的验证部分
valid_dataset = WikiDataset(valid_iter, vocab)  # 创建验证数据集

# 创建验证数据加载器
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
```

### 验证与保存最佳模型

```python
import os  # 导入os库 

min_valid_loss = float("inf")  # 初始化最低验证损失为无穷大
save_path = "best_model.pth"   # 设置模型保存路径

for epoch in range(epochs):
    # ……训练代码（见上文）
    
    # 评估模型
    model.eval()  # 将模型设置为评估模式
    valid_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for source, target in valid_dataloader:
            inputs, targets = source.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            valid_loss += loss.item()
    valid_loss /= len(valid_dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {valid_loss}")
    
    # 保存损失最小的模型
    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with validation loss {valid_loss:.4f}")
```

**说明：**
- `model.eval()`：关闭dropout等训练专用层
- 用`torch.no_grad()`节省显存、加速推理
- 只保存验证损失最小的模型，防止过拟合
- 可用`model.load_state_dict(torch.load(save_path))`恢复最佳权重

---

## 文本生成（Beam Search）

### 集束搜索生成函数

```python
# 定义集束搜索的函数
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval()  # 将模型设置为评估模式
    # 将输入字符串中的每个token转换为其在词汇表中的索引
    input_tokens = [vocab[token] for token in input_str.split()]
    # 创建一个列表，用于存储候选序列
    candidates = [(input_tokens, 0.0)]
    
    with torch.no_grad():  # 禁用梯度计算
        for _ in range(max_len):  # 生成最多max_len个token
            new_candidates = []
            for candidate, candidate_score in candidates:
                inputs = torch.LongTensor(candidate).unsqueeze(0).to(device)
                outputs = model(inputs)  # 输出logits形状为[1, len(output_tokens), vocab_size]
                logits = outputs[:, -1, :]  # 只关心最后一个时间步的logits
                
                # 找到具有最高分数的前beam_width个token
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = []  # 初始化输出序列
                
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    new_candidate = candidate + [next_token.item()]
                    new_score = candidate_score - score.item()  # 使用负数，因为需要降序排列
                    
                    if next_token.item() == vocab["<eos>"]:
                        # 如果生成的token是EOS（结束符），将其添加到最终结果中
                        final_results.append((new_candidate, new_score))
                    else:
                        # 将新生成的候选序列添加到新候选列表中
                        new_candidates.append((new_candidate, new_score))
            
            # 从新候选列表中选择得分最高的beam_width个序列
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]
    
    # 选择得分最高的候选序列
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]
    # 将输出的token转换回文本字符串
    output_str = " ".join([vocab.get_itos()[token] for token in best_candidate if vocab.get_itos()[token] != "<pad>"])
    return output_str

# 使用示例
model.load_state_dict(torch.load('best_model.pth'))  # 加载模型
input_str = "my name"  # 输入几个词
generated_text = generate_text_beam_search(model, input_str)  # 模型根据这些词生成后续文本
print("生成的文本：", generated_text)  # 打印生成的文本
```

### 输出示例

**可能的输出：**
```
生成的文本：my name is john . <eos>
```
或
```
生成的文本：my name was also used in 1897 by lucasfilm games in the common by lucasfilm games ...
```

### 关于重复输出的说明

如果输出出现重复片段（如"by lucasfilm games in the common"反复出现），说明：

1. **训练轮数较少**：模型还没充分学习到长距离依赖和句子终止的规律。
2. **beam search宽度较大**：可能导致模型偏向高概率但重复的片段。
3. **缺乏惩罚机制**：没有对重复token或重复片段做惩罚（如重复惩罚、温度采样等）。

**改进建议：**
- 增加训练轮数或训练数据量
- 尝试beam search时加"重复惩罚"或"温度采样"
- 训练时适当调整模型结构或正则化参数

---

## 总结

### 完整流程

```
WikiText2语料库
    ↓
分词 + 词表构建 (tokenizer + vocab)
    ↓
WikiDataset (source/target分离，自回归格式)
    ↓
DataLoader + collate_fn (批量pad，高效加载)
    ↓
GPT模型 (Decoder + Projection)
    ↓
训练循环 (前向→损失→反向→优化)
    ↓
验证 + 保存最佳模型
    ↓
Beam Search文本生成
    ↓
生成的文本
```

### 每一步的作用

| 步骤 | 作用 |
|------|------|
| 数据加载 | 获取大规模英文文本语料 |
| 分词与词表 | 将文本标准化为token索引序列 |
| Dataset | 定义单条数据的读取规则（source/target分离） |
| DataLoader | 批量采样、自动pad、打乱顺序 |
| GPT模型 | 自回归语言建模（masked self-attention） |
| 训练循环 | 持续优化参数，学习"前n-1预测第n个token" |
| 验证与保存 | 监控泛化能力，保存最佳模型 |
| 文本生成 | 用beam search自回归生成连贯文本 |

### 一句话总结

使用WikiText2大规模语料训练GPT模型，通过标准的Dataset/DataLoader管道实现高效批量训练，最终用beam search进行自回归文本生成。每一步都是为"让模型学会用历史token预测下一个token"服务。
