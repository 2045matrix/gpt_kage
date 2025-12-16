# 用 Hugging Face 预训练 GPT 微调 ChatGPT 完整指南

## 目录
- [用 Hugging Face 预训练 GPT 微调 ChatGPT 完整指南](#用-hugging-face-预训练-gpt-微调-chatgpt-完整指南)
  - [目录](#目录)
  - [安装 Hugging Face Transformers 库](#安装-hugging-face-transformers-库)
  - [第2步：载入预训练 GPT-2 模型和分词器](#第2步载入预训练-gpt-2-模型和分词器)
    - [输出示例](#输出示例)
  - [ChatDataset 类实现](#chatdataset-类实现)
    - [ChatDataset 说明](#chatdataset-说明)
  - [数据示例](#数据示例)
    - [示例 1](#示例-1)
    - [示例 2](#示例-2)
  - [DataLoader 批处理与 Padding](#dataloader-批处理与-padding)
    - [DataLoader 说明](#dataloader-说明)
    - [输出示例](#输出示例-1)
  - [模型训练](#模型训练)
    - [训练说明](#训练说明)
  - [集束搜索生成](#集束搜索生成)
    - [集束搜索说明](#集束搜索说明)
    - [生成结果示例](#生成结果示例)
  - [分词器的 BPE 原理](#分词器的-bpe-原理)
    - [分词示例](#分词示例)
    - [BPE 机制](#bpe-机制)
  - [禁止梯度计算](#禁止梯度计算)
    - [作用](#作用)
    - [使用场景](#使用场景)
    - [代码示例](#代码示例)
  - [总结](#总结)

---

## 安装 Hugging Face Transformers 库

在终端执行（建议加上 pip 升级）：

```bash
pip install --upgrade transformers
pip install --upgrade torch
```

---

## 第2步：载入预训练 GPT-2 模型和分词器

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2"  # 可选 "gpt2-medium"、"gpt2-large" 等
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 设置 pad_token，GPT-2 默认没有 pad_token
tokenizer.pad_token = '<pad>'
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 查看模型和分词器信息
vocab = tokenizer.get_vocab()
print("模型信息：", model)
print("分词器信息：", tokenizer)
print("词汇表大小：", len(vocab))
print("部分词汇示例：", list(vocab.keys())[8000:8005])
```

### 输出示例

- 模型信息：显示 GPT2LMHeadModel 结构
- 分词器信息：显示分词器参数和特殊 token
- 词汇表大小：50257
- 部分词汇示例：['parent', 'Art', 'pack', 'diplom', 'rets']

---

## ChatDataset 类实现

```python
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer, vocab):
        self.tokenizer = tokenizer  # 分词器
        self.vocab = vocab  # 词汇表
        # 加载数据并处理，将处理后的输入数据和目标数据赋值给input_data和target_data
        self.input_data, self.target_data = self.load_and_process_data(file_path)
    
    def load_and_process_data(self, file_path):        
        with open(file_path, "r") as f:  # 读取文件内容
            lines = f.readlines()
        input_data, target_data = [], []        
        for i, line in enumerate(lines):  # 遍历文件的每一行            
            if line.startswith("User:"):  # 如以"User:"开头，移除"User: "前缀，并将张量转换为列表
                tokens = self.tokenizer(line.strip()[6:], return_tensors="pt")["input_ids"].tolist()[0]
                tokens = tokens + [tokenizer.eos_token_id]  # 添加结束符
                input_data.append(torch.tensor(tokens, dtype=torch.long))  # 添加input_data
            elif line.startswith("AI:"):  # 如以"AI:"开头，移除"AI: "前缀，并将张量转换为列表
                tokens = self.tokenizer(line.strip()[4:], return_tensors="pt")["input_ids"].tolist()[0]
                tokens = tokens + [tokenizer.eos_token_id]  # 添加结束符
                target_data.append(torch.tensor(tokens, dtype=torch.long))  # 添加target_data
        return input_data, target_data
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

file_path = "chat.txt"  # 加载chat.txt数据集
chat_dataset = ChatDataset(file_path, tokenizer, vocab)  # 创建ChatDataset对象，传入文件、分词器和词汇表
for i in range(2):  # 打印数据集中前2个数据示例
    input_example, target_example = chat_dataset[i]
    print(f"示例 {i + 1}: ")
    print("输入：", tokenizer.decode(input_example))
    print("输出：", tokenizer.decode(target_example))
```

### ChatDataset 说明
- 继承自 torch.utils.data.Dataset，方便与 DataLoader 配合批量训练。
- __init__：传入文件路径、分词器、词表，调用 load_and_process_data 读取和处理数据。
- load_and_process_data：
  - 读取 chat.txt，每行以 "User:" 或 "AI:" 开头。
  - "User:" 行去掉前缀后用分词器编码，末尾加上 <eos>（结束符），存入 input_data。
  - "AI:" 行同理，处理后存入 target_data。
  - 每条 input/target 都是 LongTensor，便于后续训练。
- __len__：返回样本数量（input_data 的长度）。
- __getitem__：返回指定索引的 input/target 对。

---

## 数据示例

### 示例 1
- 输入：hi, how are you?<|endoftext|>
- 输出：i am doing well, thank you. how about you?<|endoftext|>

### 示例 2
- 输入：i am good, thanks for asking. what can you do?<|endoftext|>
- 输出：i am an ai language model. i can help you answer questions.<|endoftext|>

**说明：**
- 每个输入是用户的一句话，输出是 AI 的回复。
- 末尾都加了 <|endoftext|>（即 tokenizer.eos_token），用于标记句子结束，帮助模型学习对话的边界。
- 这种格式适合用来微调 GPT-2/ChatGPT，让模型学会"看到用户输入→生成合理回复"。

---

## DataLoader 批处理与 Padding

```python
from torch.utils.data import DataLoader
import torch

tokenizer.pad_token = '<pad>'  # 为分词器添加pad token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

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

# 定义collate_fn函数，用于将一个批次的数据整理成适当的形状
def collate_fn(batch):
    # 从批次中分离源序列和目标序列
    sources, targets = zip(*batch)    
    # 计算批次中的最大序列长度
    max_length = max(max(len(s) for s in sources), max(len(t) for t in targets))
    # 使用pad_sequence函数补齐源序列和目标序列
    sources = pad_sequence(sources, padding_value=tokenizer.pad_token_id, length=max_length)
    targets = pad_sequence(targets, padding_value=tokenizer.pad_token_id, length=max_length)    
    # 返回补齐后的源序列和目标序列
    return sources, targets

# 创建DataLoader
chat_dataloader = DataLoader(chat_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 检查Dataloader输出
for input_batch, target_batch in chat_dataloader:
     print("Input batch tensor size:", input_batch.size())
     print("Target batch tensor size:", target_batch.size())
     break

for input_batch, target_batch in chat_dataloader:
     print("Input batch tensor:")
     print(input_batch)
     print("Target batch tensor:")
     print(target_batch)
     break
```

### DataLoader 说明
1. **pad_sequence**：将不同长度的序列补齐到同一长度（用 pad_token_id 填充）。返回 shape：[batch_size, max_seq_len]

2. **collate_fn**：自定义 DataLoader 的批处理方式。
   - 拆分 batch 为 sources（输入）和 targets（输出）。
   - 计算本批次最大长度，分别对 sources 和 targets 补齐。
   - 返回补齐后的张量。

3. **DataLoader**：用法 `chat_dataloader = DataLoader(chat_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)`
   - 每个 batch 输出：input_batch/target_batch shape: [batch_size, max_seq_len]
   - 数据已自动 padding，适合送入模型训练。

### 输出示例

Input batch tensor:
```
tensor([[   72,   716,   922,   837,  5176,   329,  4737,   764,   644,   460,    345,   466,  5633, 50256, 50256, 50256],
        [40716,   345,  5145,  1312,   481,  2198,   340,   503,   764, 50256,  50256, 50256, 50256, 50256, 50256, 50256]])
```

Target batch tensor:
```
tensor([[   72,   716,   281,   257,    72,  3303,  2746,   764,  1312,   460,  1037,   345,  3280,  2683,   764, 50256],
        [ 5832,   389,  7062,  5145,  1309,   502,   760,   611,   345,   761,  1037,   351,  1997,  2073,   764, 50256]])
```

---

## 模型训练

```python
import torch.nn as nn
import torch.optim as optim

# 定义损失函数，忽略pad_token_id对应的损失值
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 进行500个epoch的训练
for epoch in range(500):    
    for batch_idx, (input_batch, target_batch) in enumerate(chat_dataloader):  # 遍历数据加载器中的批次       
        optimizer.zero_grad()  # 梯度清零   
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标批次移至设备
        outputs = model(input_batch)  # 前向传播
        logits = outputs.logits  # 获取logits        
        loss = criterion(logits.view(-1, len(vocab)), target_batch.view(-1))  # 计算损失
        loss.backward()  # 反向传播        
        optimizer.step()  # 更新参数    
    if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次损失值
        print(f'Epoch: {epoch + 1:04d}, cost = {loss:.6f}')
```

### 训练说明
1. 损失函数：nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) —— 忽略 padding 部分的损失，保证模型只在有效token上学习。

2. 优化器：optimizer = optim.Adam(model.parameters(), lr=0.0001)

3. 训练循环：
   - 遍历 DataLoader，批量送入模型。
   - 前向传播，outputs = model(input_batch)
   - 取 logits（outputs.logits），与 target_batch 计算损失。
   - loss.backward() + optimizer.step() 完成参数更新。
   - 每100个epoch打印一次损失。

4. 设备适配：input_batch, target_batch = input_batch.to(device), target_batch.to(device)

---

## 集束搜索生成

```python
# 定义集束解码函数
def generate_text_beam_search(model, input_str, max_len=50, beam_width=5):
    model.eval()  # 将模型设置为评估模式（不计算梯度）
    # 对输入字符串进行编码，并将其转换为张量，然后将其移动到相应的设备上
    input_tokens = tokenizer.encode(input_str, return_tensors="pt").to(device)    
    # 初始化候选序列列表，包含当前输入序列和其对数概率得分（我们从0开始）
    candidates = [(input_tokens, 0.0)]    
    # 禁用梯度计算，以加速预测过程 
    with torch.no_grad():
        # 迭代生成最大长度的序列
        for _ in range(max_len):
            new_candidates = []
            # 对于每个候选序列
            for candidate, candidate_score in candidates:
                # 使用模型进行预测
                outputs = model(candidate)
                # 获取输出 logits
                logits = outputs.logits[:, -1, :]
                # 获取对数概率得分的 top-k 值（即 beam_width）及其对应的 token
                scores, next_tokens = torch.topk(logits, beam_width, dim=-1)
                final_results = []
                # 遍历 top-k token 及其对应的得分
                for score, next_token in zip(scores.squeeze(), next_tokens.squeeze()):
                    # 在当前候选序列中添加新的 token
                    new_candidate = torch.cat((candidate,next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
                    # 更新候选序列的得分
                    new_score = candidate_score - score.item()                    
                    # 如果新的 token 是结束符(eos_token)，则将该候选序列添加到最终结果中
                    if next_token.item() == tokenizer.eos_token_id:
                        final_results.append((new_candidate, new_score))
                    # 否则，将新的候选序列添加到新候选序列列表中
                    else:
                        new_candidates.append((new_candidate, new_score))
            # 从新候选序列列表中选择得分最高的 top-k 个序列
            candidates = sorted(new_candidates, key=lambda x: x[1])[:beam_width]    
    # 选择得分最高的候选序列
    best_candidate, _ = sorted(candidates, key=lambda x: x[1])[0]    
    # 将输出 token 转换回文本字符串
    output_str = tokenizer.decode(best_candidate[0])    
    # 移除输入字符串并修复空格问题
    input_len = len(tokenizer.encode(input_str))
    output_str = tokenizer.decode(best_candidate.squeeze()[input_len:])    
    return output_str

# 测试模型
test_inputs = [
    "what is the weather like today?",
    "can you recommend a good book?"]

# 输出测试结果
for i, input_str in enumerate(test_inputs, start=1):
    generated_text = generate_text_beam_search(model, input_str)
    print(f"测试 {i}:")
    print(f"User: {input_str}")
    print(f"AI: {generated_text}")
```

### 集束搜索说明
- 主要流程：每步扩展所有候选序列，取top-k（beam_width）概率最高的token，生成新候选。
- 若遇到eos_token，提前终止并收集结果。
- 每轮只保留得分最高的beam_width个候选，循环max_len步。
- 最终输出得分最高的序列，去除输入部分，仅保留生成内容。

### 生成结果示例

测试 1:
User: what is the weather like today?<|endoftext|>
AI: you need an current time for now app with app app app app

测试 2:
User: Can you recommend a good book?<|endoftext|>
AI: ockingbird Lee Harper Harper Taylor

---

## 分词器的 BPE 原理

### 分词示例

```python
tokenizer.encode('Mockingbird'): [44, 8629, 16944]
tokenizer.decode(44): 'M'
tokenizer.decode(8629): 'ocking'
tokenizer.decode(16944): 'bird'
```

**说明：**
- tokenizer.encode('Mockingbird') 得到 [44, 8629, 16944]
- tokenizer.decode(44) 得到 'M'
- tokenizer.decode(8629) 得到 'ocking'
- tokenizer.decode(16944) 得到 'bird'

即：'Mockingbird' 被拆成 'M' + 'ocking' + 'bird' 三个子词（token）。

### BPE 机制
这就是BPE（Byte Pair Encoding）分词的典型表现，能处理未登录词和长词，提升模型泛化能力。

---

## 禁止梯度计算

禁止梯度计算（如 `with torch.no_grad():`）是指在推理/生成等阶段，不记录和计算反向传播所需的梯度信息。

### 作用
1. **节省显存和计算资源**：不保存中间梯度。
2. **加快推理速度**：减少计算开销。
3. **防止误用反向传播**：不会影响模型参数。

### 使用场景
常用于模型评估、生成文本、inference等只需前向推理、不需训练的场景。

### 代码示例

```python
with torch.no_grad():
    # 禁用梯度计算的代码块
    for _ in range(max_len):
        inputs = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
        outputs = model(inputs)
        logits = outputs.logits[:, -1, :]
        _, next_token = torch.max(logits, dim=-1)
        next_token = next_token.item()
        output_tokens.append(next_token)
```

---

## 总结

整个微调流程包括：
1. 安装 Transformers 库
2. 加载预训练 GPT-2 模型和分词器
3. 创建 ChatDataset 处理对话数据
4. 用 DataLoader 进行批处理和 padding
5. 编写训练循环，用交叉熵损失优化参数
6. 用集束搜索生成高质量对话回复
7. 理解 BPE 分词和梯度计算的作用

这套完整流程可用于微调 GPT-2/ChatGPT，实现对话模型的端到端开发。

