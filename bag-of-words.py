# Bag-of-Words
# 构建一个玩具数据集
corpus = ["我特别特别喜欢看电影",
        "这部电影真的是很好看的电影",
        "今天天气真好是难得的好天气",
        "我今天去看了一部电影",
        "电影院的电影都很好看"]

# 对句子进行分词
import jieba # 导入jieba包
# 使用jieba.cut进行分词，并将结果转换为列表，存储在corpus_tokenized中
corpus_tokenized = [list(jieba.cut(sentence)) for sentence in corpus]

# 创建词汇表
word_dict = {} # 初始化词汇表
# 遍历分词后的语料库
for sentence in corpus_tokenized:
    for word in sentence:
        # 如果词汇表中没有该词，则将其添加到词汇表中
        if word not in word_dict:
            word_dict[word] = len(word_dict) #分配当前词汇表索引
print("词汇表：", word_dict) # 打印词汇表

# 根据词汇表将句子转换为词袋表示
bow_vectors = [] # 初始化词袋表示
# 遍历分词后的语料库
for sentence in corpus_tokenized:
    # 初始化一个全0向量，其长度等于词汇表大小
    sentence_vector = [0] * len(word_dict)
    for word in sentence:
        # 给对应词索引位置上的数加1，表示该词在当前句子中出现了一次
        sentence_vector[word_dict[word]] += 1
    # 将当前句子的词袋向量添加到向量列表中
    bow_vectors.append(sentence_vector)
print("词袋表示：", bow_vectors) # 

# 导入numpy库，用于计算余弦相似度
import numpy as np 
# 定义余弦相似度函数
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2) # 计算向量vec1和vec2的点积
    norm_a = np.linalg.norm(vec1) # 计算向量vec1的范数
    norm_b = np.linalg.norm(vec2) # 计算向量vec2的范数   
    return dot_product / (norm_a * norm_b) # 返回余弦相似度
# 初始化一个全0矩阵，用于存储余弦相似度
similarity_matrix = np.zeros((len(corpus), len(corpus)))
# 计算每两个句子之间的余弦相似度
for i in range(len(corpus)):
    for j in range(len(corpus)):
        similarity_matrix[i][j] = cosine_similarity(bow_vectors[i], 
                                                    bow_vectors[j])

# 导入matplotlib库，用于可视化余弦相似度矩阵
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 指定中文字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
custom_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
fig, ax = plt.subplots() # 创建一个绘图对象
# 使用matshow函数绘制余弦相似度矩阵，颜色使用蓝色调
cax = ax.matshow(similarity_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax) # 条形图颜色映射
ax.set_xticks(range(len(corpus))) # x轴刻度
ax.set_yticks(range(len(corpus))) # y轴刻度
ax.set_xticklabels(corpus, rotation=45, ha='left', fontproperties=custom_font) # 刻度标签 
ax.set_yticklabels(corpus, fontproperties=custom_font) # 刻度标签为原始句子
plt.savefig("similarity_matrix.png", bbox_inches='tight') # 保存为PNG图片
print("相似度矩阵图已保存为 similarity_matrix.png")
