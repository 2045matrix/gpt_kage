# Wiki-GPT对话微调完整指南

## 目录
1. [项目概述](#项目概述)
2. [数据准备](#数据准备)
3. [模型加载](#模型加载)
4. [微调训练](#微调训练)
5. [参数冻结策略](#参数冻结策略)
6. [文本生成](#文本生成)
7. [结果分析](#结果分析)
8. [改进建议](#改进建议)

---

## 项目概述

本项目的目标是：**将预训练的Wiki-GPT（基于WikiText2数据集）通过对话数据集（chat.txt）进行微调，赋予模型对话生成能力，最终实现ChatGPT式的对话功能**。

### 核心思路

- **预训练基础**：Wiki-GPT已在大规模文本上学习了语言表达能力（词汇、语法、知识）。
- **对话微调**：用User-AI对话对进行微调，让模型学会根据用户输入生成合适回复。
- **参数高效**：冻结低层参数，只微调高层参数，保留预训练知识同时适应对话任务。

---

## 数据准备

### ChatDataset类：对话数据加载

```python
import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    """
    对话数据集类，用于加载chat.txt中的User-AI对话对
    输入输出格式：
    - Input: User问题（加<sos>和<eos>，转为token索引）
    - Target: AI回答（加<sos>和<eos>，转为token索引）
    """
    def __init__(self, file_path, tokenizer, vocab):
        self.tokenizer = tokenizer  # 分词器
        self.vocab = vocab          # 词汇表
        self.input_data, self.target_data = self.load_and_process_data(file_path)
    
    def load_and_process_data(self, file_path):
        """
        读取chat.txt，按User:/AI:标记分离输入输出
        chat.txt格式示例：
        User: hi , how are you ?
        AI: i am fine thank you .
        User: what is the weather like today ?
        AI: sunny and warm .
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        input_data, target_data = [], []
        for i, line in enumerate(lines):
            if line.startswith("User:"):  # 提取User问题
                tokens = self.tokenizer(line.strip()[6:])  # 去掉"User: "前缀
                tokens = ["<sos>"] + tokens + ["<eos>"]
                indices = [self.vocab[token] for token in tokens]
                input_data.append(torch.tensor(indices, dtype=torch.long))
            elif line.startswith("AI:"):   # 提取AI回答
                tokens = self.tokenizer(line.strip()[4:])  # 去掉"AI: "前缀
                tokens = ["<sos>"] + tokens + ["<eos>"]
                indices = [self.vocab[token] for token in tokens]
                target_data.append(torch.tensor(indices, dtype=torch.long))
        
        return input_data, target_data
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

# 使用示例
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
train_iter = WikiText2(split='train')

def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                  specials=["<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<pad>"])

file_path = "chat.txt"  # User-AI对话文件
chat_dataset = ChatDataset(file_path, tokenizer, vocab)

# 打印几个样本
for i in range(3):
    input_sample, target_sample = chat_dataset[i]
    print(f"Sample {i + 1}:")
    print("Input Data: ", input_sample)
    print("Target Data: ", target_sample)
    print("-" * 50)
```

### DataLoader和批处理

```python
from torch.utils.data import DataLoader

def pad_sequence(sequences, padding_value=0, length=None):
    """
    将不同长度的序列补齐到相同长度，便于批量处理
    - 如果length为None，补齐到batch中的最大长度
    - 用padding_value（通常是<pad>的索引0）填充
    """
    max_length = max(len(seq) for seq in sequences) if length is None else length
    result = torch.full((len(sequences), max_length), padding_value, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        end = len(seq)
        result[i, :end] = seq[:end]
    
    return result

def collate_fn(batch):
    """
    自定义DataLoader的batch整理函数
    - 从batch中分离input和target
    - 对两者分别调用pad_sequence补齐
    - 返回补齐后的张量对
    """
    sources, targets = zip(*batch)
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    sources = pad_sequence(sources, padding_value=vocab["<pad>"], length=max_length)
    targets = pad_sequence(targets, padding_value=vocab["<pad>"], length=max_length)
    
    return sources, targets

# 创建DataLoader
batch_size = 2
chat_dataloader = DataLoader(chat_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
```

---

## 模型加载

### 加载预训练Wiki-GPT

```python
from GPT_Model import GPT  # 导入自定义的GPT模型类
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建模型实例
# vocab_size: 28785（WikiText2词汇表大小）
# max_seq_len: 256（最大序列长度）
# n_layers: 6（Decoder层数）
model = GPT(28785, 256, n_layers=6).to(device)

# 加载预训练权重（从wiki_gpt.md训练得到）
model.load_state_dict(torch.load('best_model.pt'))

print("模型结构：")
print(model)
print("\n模型已加载到", device)

# 验证所有参数都被正确加载
# 输出：<All keys matched successfully>
```

模型结构示例：
```
GPT(
  (decoder): Decoder(
    (src_emb): Embedding(28785, 512)
    (pos_emb): Embedding(256, 512)
    (layers): ModuleList(
      (0-5): 6个DecoderLayer，每个包含：
        (self_attn): MultiHeadAttention
        (feed_forward): PoswiseFeedForwardNet
    )
  )
  (projection): Linear(in_features=512, out_features=28785)
)
```

---

## 微调训练

### 完整的微调训练循环

```python
import torch.nn as nn
import torch.optim as optim

# 定义损失函数（忽略<pad>位置的损失）
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

# 定义优化器（Adam）
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 微调参数
num_epochs = 100
log_interval = 20

# 训练循环
for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx, (input_batch, target_batch) in enumerate(chat_dataloader):
        # 梯度清零
        optimizer.zero_grad()
        
        # 移动到GPU/CPU
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        
        # 前向传播
        # input_batch: [batch_size, seq_len]
        # outputs: [batch_size, seq_len, vocab_size]
        outputs = model(input_batch)
        
        # 计算损失
        # 展平为[batch_size * seq_len, vocab_size]和[batch_size * seq_len]
        # CrossEntropyLoss会自动忽略ignore_index的位置
        loss = criterion(outputs.view(-1, len(vocab)), target_batch.view(-1))
        
        # 累计损失
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
    
    # 定期打印损失
    if (epoch + 1) % log_interval == 0:
        avg_loss = total_loss / len(chat_dataloader)
        print(f"Epoch: {epoch + 1:03d}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # （可选）保存最优模型
    # if avg_loss < best_loss:
    #     best_loss = avg_loss
    #     torch.save(model.state_dict(), 'chat_gpt_best.pt')

print("微调完成！")
```

### 训练要点

1. **CrossEntropyLoss(ignore_index=vocab["<pad>"])**：自动忽略padding位置，避免对无意义token的梯度更新。
2. **outputs.view(-1, len(vocab))**：将[batch_size, seq_len, vocab_size]展平为[batch_size*seq_len, vocab_size]以匹配损失函数要求。
3. **lr=0.0001**：较小的学习率，避免破坏预训练权重。
4. **Adam优化器**：自适应学习率，适合微调任务。

---

## 参数冻结策略

### 冻结低层参数，只微调高层

```python
def freeze_decoder_layers(model, n):
    """
    冻结Decoder的前n层参数，只让后面层和输出层参与更新
    
    原理：
    - 前n层学习了通用的语言表达（词序、语法等），来自预训练
    - 后面层学习任务特定的知识（对话风格）
    - 冻结前n层可以稳定训练，减少过拟合
    """
    for i, layer in enumerate(model.decoder.layers):
        if i < n:
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Layer {i} frozen")

# 冻结前2层
freeze_decoder_layers(model, n=2)

# 只优化未冻结的参数
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=0.0001)

# 验证：计算可训练参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数：{total_params:,}")
print(f"可训练参数数：{trainable:,}")
print(f"比例：{100 * trainable / total_params:.2f}%")
```

### 为什么冻结参数？

| 策略 | 优点 | 缺点 |
|-----|-----|-----|
| **全量微调** | 充分学习对话风格 | 容易过拟合，损坏预训练知识 |
| **冻结低层** | 稳定，快速收敛，保留预训练 | 表达能力受限 |
| **冻结高层** | 保留任务特定知识 | 通常不合理，低层更通用 |

对话微调通常采用**冻结低层**策略，因为：
- 低层（词嵌入、基础语法）是通用的，预训练已充分学习
- 高层（语义理解、生成策略）是任务特定的，需要微调

---

## 文本生成

### 贪心采样

```python
def generate_text_greedy(model, input_tokens, max_len=50):
    """
    使用贪心采样生成文本（速度快，但多样性低）
    每次选择概率最高的token，直到生成<eos>或达到max_len
    """
    model.eval()
    device = next(model.parameters()).device
    
    # input_tokens: 分词后的起始序列，如["hi", ",", "how", "are", "you", "?"]
    input_ids = [vocab["<sos>"]] + [vocab.get(tok, vocab["<pad>"]) for tok in input_tokens]
    output_tokens = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
            outputs = model(inputs)  # [1, seq_len, vocab_size]
            
            # 取最后一个位置的最大概率token（贪心）
            _, next_token = torch.max(outputs[:, -1, :], dim=-1)
            next_token = next_token.item()
            
            if next_token == vocab["<eos>"]:
                break
            
            output_tokens.append(next_token)
    
    # 还原为文本
    output_str = " ".join([corpus.idx2word[idx] for idx in output_tokens 
                          if idx not in (vocab["<sos>"], vocab["<eos>"], vocab["<pad>"])])
    return output_str

# 使用示例
input_text = "hi , how are you ?"
generated = generate_text_greedy(model, input_text.split())
print("Generated:", generated)
```

### Beam Search（推荐）

```python
def generate_text_beam_search(model, input_tokens, max_len=50, beam_width=5):
    """
    使用Beam Search生成文本（多样性好，质量高）
    
    原理：
    - 维护beam_width个候选序列
    - 每步扩展每个序列，选择分数最高的beam_width个继续
    - 直到所有序列都生成<eos>或达到max_len
    - 返回分数最高的序列
    """
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = [vocab["<sos>"]] + [vocab.get(tok, vocab["<pad>"]) for tok in input_tokens]
    # beams: (token_id序列, 累积log概率)
    beams = [(input_ids, 0.0)]
    completed = []
    
    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                # 如果序列已完成，保存到completed
                if seq[-1] == vocab["<eos>"]:
                    completed.append((seq, score))
                    continue
                
                # 预测下一个token
                input_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
                output = model(input_tensor)  # [1, seq_len, vocab_size]
                logits = output[0, -1, :]     # 取最后一个位置
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 取分数最高的beam_width个token
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for log_p, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + log_p
                    new_beams.append((new_seq, new_score))
            
            # 只保留分数最高的beam_width个序列（剪枝）
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams
            
            # 如果所有序列都已完成，停止
            if all(seq[-1] == vocab["<eos>"] for seq, _ in beams):
                break
        
        # 合并已完成和未完成的序列
        completed += [b for b in beams if b[0][-1] == vocab["<eos>"]]
        if not completed:
            completed = beams
        
        # 选分数最高的序列
        best_seq = max(completed, key=lambda x: x[1])[0]
        
        # 还原为文本
        tokens = [corpus.idx2word[idx] for idx in best_seq 
                 if idx not in (vocab["<sos>"], vocab["<eos>"], vocab["<pad>"])]
        return " ".join(tokens)

# 使用示例
input_text = "hi , how are you ?"
generated = generate_text_beam_search(model, input_text.split(), beam_width=5)
print("Generated:", generated)
```

### 生成方法对比

| 方法 | 速度 | 多样性 | 质量 | 用途 |
|-----|------|--------|------|-----|
| **贪心** | 最快 | 低 | 一般 | 快速推理 |
| **Beam Search** | 中等 | 中-高 | 高 | 生产环境 |
| **Top-K采样** | 快 | 高 | 一般 | 对话/创意任务 |
| **Top-P采样** | 快 | 中-高 | 好 | 平衡多样性和质量 |

---

## 结果分析

### 生成示例

```
Input: "hi , how are you ?"

Greedy输出:
"hi , how are you ? i am fine thank you . how are you ?"

Beam Search (beam_width=5) 输出:
"hi , how are you ? i am fine thank you . i am a student ."

预期输出:
"hi , how are you ? i am fine thank you . how are you doing today ?"
```

### 问题分析

生成结果中出现的重复（如"you , ai you , ai you ..."）原因：

1. **训练数据量不足**
   - chat.txt样本太少，模型未充分学习对话多样性
   - 解决：增加对话数据，丰富样本

2. **模型容量与数据不匹配**
   - 6层Decoder处理小数据集易过拟合
   - 解决：增加数据或降低模型复杂度

3. **Beam Search配置不当**
   - beam_width=5太大，容易保留重复分支
   - 解决：降低beam_width（如3），加入重复惩罚

4. **缺乏长期上下文建模**
   - max_seq_len=256可能不够长
   - 解决：支持更长序列或添加对话历史

---

## 改进建议

### 1. 数据质量改进

```python
# chat.txt增强：多样化和充分的对话数据
# 示例：至少100个User-AI对话对
User: what is your name ?
AI: i am an assistant named chatgpt .
User: can you help me with programming ?
AI: of course ! i can help you with python , java , javascript and more .
User: how do you learn ?
AI: i learn through fine-tuning on large dialogue datasets .
...（更多对话）
```

### 2. 避免重复的采样策略

```python
def generate_with_repetition_penalty(model, input_tokens, max_len=50, 
                                     penalty=1.2, beam_width=3):
    """
    加入重复惩罚：降低已出现token的概率
    penalty > 1时，重复token的分数会降低
    """
    # 在log_probs中减去已出现token的惩罚
    for token_idx in already_generated_tokens:
        log_probs[token_idx] -= torch.log(torch.tensor(penalty))
    # ...rest of beam search
    pass
```

### 3. 训练策略优化

```python
# 关键参数调整
num_epochs = 200          # 增加训练轮数
batch_size = 4            # 增加batch size（如果显存允许）
lr = 0.00005              # 降低学习率，更稳定
warmup_steps = 100        # 添加学习率热身
gradient_accumulation = 2 # 梯度累积，模拟更大batch

# 早停机制
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # ...训练代码
    val_loss = evaluate(model, val_dataloader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_chat_gpt.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

### 4. 生成效果改进清单

- [ ] 增加chat.txt数据到500+对话对
- [ ] 使用top-p采样替代beam search
- [ ] 添加重复惩罚机制
- [ ] 实现对话历史context管理
- [ ] 添加特殊token：[USER]、[AI]、[SEP]标记角色
- [ ] 评估BLEU、ROUGE、困惑度等指标
- [ ] 人工评估生成质量

---

## 总结

### 核心流程

```
预训练Wiki-GPT (best_model.pt)
    ↓
加载模型到GPU
    ↓
准备对话数据 (chat.txt → ChatDataset → DataLoader)
    ↓
冻结低层参数，定义优化器
    ↓
微调训练（100 epochs）
    ↓
评估与推理（Beam Search生成）
    ↓
ChatGPT式对话模型
```

### 关键指标

- **预训练词汇**：28,785词（WikiText2）
- **模型大小**：6层Decoder，512维嵌入 ≈ 100M参数
- **微调数据**：chat.txt中User-AI对话对
- **生成策略**：Beam Search（beam_width=3-5）

### 进一步优化方向

1. **多轮对话**：维护对话历史，支持context-aware生成
2. **知识融合**：引入检索增强生成（RAG）
3. **风格控制**：条件生成，控制回复长度、风格
4. **实时评估**：在线A/B测试，持续改进

---

**项目完成！你已成功构建了基于Wiki-GPT的对话AI。继续积累更多高质量对话数据，模型效果会显著提升。**
