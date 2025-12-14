# Skip-Gram 示例实现

# 定义一个句子列表，后面会用这些句子来训练CBOW和Skip-Gram模型
sentences = ["Kage is Teacher", "Mazong is Boss", "Niuzong is Boss",
             "Xiaobing is Student", "Xiaoxue is Student",]
# 将所有句子连接在一起，然后用空格分隔成多个单词
words = '  '.join(sentences).split()
# 构建词汇表，去除重复的词
word_list = list(set(words))
# 创建一个字典，将每个词映射到一个唯一的索引
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
# 创建一个字典，将每个索引映射到对应的词
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
voc_size = len(word_list) # 计算词汇表的大小
print("词汇表：", word_list) # 输出词汇表
print("词汇到索引的字典：", word_to_idx) # 输出词汇到索引的字典
print("索引到词汇的字典：", idx_to_word) # 输出索引到词汇的字典
print("词汇表大小：", voc_size) # 输出词汇表大小

# 生成Skip-Gram训练数据
def create_skipgram_dataset(sentences, window_size=2):
    data = [] # 初始化数据
    for sentence in sentences: # 遍历句子
        sentence = sentence.split()  # 将句子分割成单词列表
        for idx, word in enumerate(sentence):  # 遍历单词及其索引
            # 获取相邻的单词，将当前单词前后各N个单词作为相邻单词
            for neighbor in sentence[max(idx - window_size, 0): 
                        min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:  # 排除当前单词本身
                    # 将相邻单词与当前单词作为一组训练数据
                    data.append((word, neighbor))
    return data
# 使用函数创建Skip-Gram训练数据
skipgram_data = create_skipgram_dataset(sentences)
# 打印未编码的Skip-Gram数据样例（前3个）
print("Skip-Gram数据样例（未编码）：", skipgram_data[:3])

# 定义One-Hot编码函数
import torch # 导入torch库
def one_hot_encoding(word, word_to_idx):    
    tensor = torch.zeros(len(word_to_idx)) # 创建一个长度与词汇表相同的全0张量  
    tensor[word_to_idx[word]] = 1  # 将对应词索引位置上的值设为1
    return tensor  # 返回生成的One-Hot编码后的向量
# 展示One-Hot编码前后的数据
word_example = "Teacher"
print("One-Hot编码前的单词：", word_example)
print("One-Hot编码后的向量：", one_hot_encoding(word_example, word_to_idx))
# 展示编码后的Skip-Gram训练数据样例
print("Skip-Gram 数据样例（已编码）：", [(one_hot_encoding(target, word_to_idx), word_to_idx[context])  for context, target in skipgram_data[:3]])
import torch.nn as nn # 导入neural network

# 定义Skip-Gram类
class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        # 从词汇表大小到嵌入层大小（维度）的线性层（权重矩阵）
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        # 从嵌入层大小（维度）到词汇表大小的线性层（权重矩阵）
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)
    def forward(self, X):  # 前向传播的方式，X形状为(batch_size, voc_size)
        # 通过隐藏层，hidden形状为 (batch_size, embedding_size)
        hidden = self.input_to_hidden(X)
        # 通过输出层，output_layer形状为 (batch_size, voc_size)
        output = self.hidden_to_output(hidden)
        return output

embedding_size = 2  # 设定嵌入层的大小，这里选择2是为了方便展示
skipgram_model = SkipGram(voc_size, embedding_size)  # 实例化Skip-Gram模型
print("Skip-Gram类：", skipgram_model)

# 打印权重矩阵
print("input_to_hidden 权重矩阵（每行是一个词的词向量）：")
print(skipgram_model.input_to_hidden.weight)
print("hidden_to_output 权重矩阵：")
print(skipgram_model.hidden_to_output.weight)

# 定义损失函数和优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，用于分类任务
optimizer = optim.SGD(skipgram_model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
epochs = 100  # 训练轮数
for epoch in range(epochs):
    total_loss = 0
    for target_word, context_word in skipgram_data:
        target_tensor = one_hot_encoding(target_word, word_to_idx)
        context_idx = word_to_idx[context_word]
        # 前向传播
        output = skipgram_model(target_tensor.unsqueeze(0))  # 添加batch维度
        # 计算损失
        loss = criterion(output, torch.tensor([context_idx]))
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("训练完成！")
print("训练后的词向量（input_to_hidden 权重矩阵的列）：")
print(skipgram_model.input_to_hidden.weight.T)  # 转置后每行是词向量

# 输出Skip-Gram习得的词嵌入
print("Skip-Gram词嵌入：")
for word, idx in word_to_idx.items():
    print(f"{word}: {skipgram_model.input_to_hidden.weight[:,idx].detach().numpy()}")

# 可视化词嵌入
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for word, idx in word_to_idx.items():
    vec = skipgram_model.input_to_hidden.weight[:,idx].detach().numpy()
    ax.scatter(vec[0], vec[1])
    ax.annotate(word, (vec[0], vec[1]), fontsize=12)
plt.title('二维词嵌入')
plt.xlabel('向量维度1')
plt.ylabel('向量维度2')
plt.show()
plt.savefig('skipgram_word_embedding.png')  # 保存为图片
