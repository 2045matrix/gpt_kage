import torch.nn as nn # 导入神经网络模块
# 定义神经概率语言模型(NPLM)
class NPLM(nn.Module):
    def __init__(self):
        super(NPLM, self).__init__() # 调用父类的构造函数
        self.C = nn.Embedding(voc_size, embedding_size) # 定义一个词嵌入层
        # 用LSTM层替代第一个线性层，其输入大小为embedding_size，隐藏层大小为n_hidden
        self.lstm = nn.LSTM(embedding_size, n_hidden, batch_first=True) 
        # 第二个线性层，其输入大小为n_hidden，输出大小为voc_size，即词汇表大小
        self.linear = nn.Linear(n_hidden, voc_size) 
    def forward(self, X):  # 定义前向传播过程
        # 输入数据X张量的形状为 [batch_size, n_step]
        X = self.C(X)  # 将X通过词嵌入层，形状变为 [batch_size, n_step, embedding_size]
        # 通过LSTM层
        lstm_out, _ = self.lstm(X) # lstm_out 形状变为 [batch_size, n_step, n_hidden]
        # 只选择最后一个时间步的输出作为全连接层的输入，通过第二个线性层得到输出 
        output = self.linear(lstm_out[:, -1, :]) # output的形状为 [batch_size, voc_size]
        return output # 返回输出结果

        