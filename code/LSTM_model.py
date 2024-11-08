# LSTM文本生成
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba
from tqdm import tqdm

# 定义字典类
class Dictionary(object):

    def __init__(self):
        self.word2idx = {}  # 单词到索引的映射
        self.idx2word = {}  # 索引到单词的映射
        self.idx = 0        # 索引计数器

    def __len__(self):
        return len(self.word2idx)

    # 添加单词到字典中
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # 为单词分配一个索引
            self.idx2word[self.idx] = word  # 保存索引到单词的映射
            self.idx += 1                   # 更新索引计数器

# 定义语料库类
class Corpus(object):

    def __init__(self):
        self.dictionary = Dictionary()

    # 读取数据文件，处理文本并返回ID表示的文本数据
    def get_data(self, path, batch_size=20):
        # 第一步：构建词汇表
        with open(path, 'r', encoding="utf-8") as f:
            tokens = 0
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']  # 对每行文本进行分词，并加上句子结束标记
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)   # 将单词加入字典中

        # 第二步：将文本转为ID
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]  # 将单词转换为对应的索引
                    token += 1

        # 第三步：按批次重新排列数据
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

# 定义LSTM模型
class LSTMmodel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 词嵌入层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM层
        self.linear = nn.Linear(hidden_size, vocab_size)  # 全连接层

    def forward(self, x, h):
        x = self.embed(x)  # 获取词嵌入表示
        out, (h, c) = self.lstm(x, h)  # 通过LSTM层
        out = out.reshape(out.size(0) * out.size(1), out.size(2))  # 重塑输出形状
        out = self.linear(out)  # 通过全连接层

        return out, (h, c)

# 定义模型参数
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 20
batch_size = 50
seq_length = 30
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建语料库并获取数据
corpus = Corpus()
ids = corpus.get_data('data/data6.txt', batch_size)
vocab_size = len(corpus.dictionary)

# 初始化模型、损失函数和优化器
model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):

    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    for i in tqdm(range(0, ids.size(1) - seq_length, seq_length)):
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)

        states = [state.detach() for state in states]  # 分离状态
        outputs, states = model(inputs, states)  # 前向传播
        loss = cost(outputs, targets.reshape(-1))  # 计算损失

        model.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪
        optimizer.step()  # 更新参数

# 生成文本
num_samples = 20000

state = (torch.zeros(num_layers, 1, hidden_size).to(device),
        torch.zeros(num_layers, 1, hidden_size).to(device))

prob = torch.ones(vocab_size)
_input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

article = []
for i in range(num_samples):
    output, state = model(_input, state)

    prob = output.exp()
    word_id = torch.multinomial(prob, num_samples=1).item()

    _input.fill_(word_id)

    word = corpus.dictionary.idx2word[word_id]
    word = '\n' if word == '<eos>' else word
    article.append(word)
    
# 保存生成的文本
with open('data/result6.txt', 'w', encoding='utf-8') as f:
    for art in article:
        f.write(art)
