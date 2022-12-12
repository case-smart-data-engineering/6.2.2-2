# !/usr/bin/env Python3
# 实现基于skip-gram算法的Word2Vec预训练语言模型
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
 
sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
sentences_list = "".join([i for i in sentences]).split()
vocab = list(set(sentences_list))
word2idx = {j: i for i, j in enumerate(vocab)}
idx2word = {i: j for i, j in enumerate(vocab)}
vocab_size = len(vocab)
window_size = 2
embedding_size = 2
 
 
def make_data(seq_data):
    context_arr = []
    center = []
    context = []
    skip_gram = []
    seq_data = "".join([i for i in seq_data]).split()
    for sen in seq_data:
        for step in range(window_size, len(sen) - window_size):
            center = step
            context_arr = list(range(step - window_size, step)) + list(range(step + 1, step + window_size))
            for context_i in context_arr:
                skip_gram.append([np.eye(vocab_size)[word2idx[seq_data[center]]], context_i])
    input_data = []
    target_data = []
    for a, b in skip_gram:
        input_data.append(a)
        target_data.append(b)
    return torch.FloatTensor(input_data), torch.LongTensor(target_data)
 
 
class my_dataset(Dataset):
    def __init__(self, input_data, target_data):
        super(my_dataset, self).__init__()
        self.input_data = input_data
        self.target_data = target_data
 
    def __getitem__(self, index):
        return self.input_data[index], self.target_data[index]
 
    def __len__(self):
        return self.input_data.size(0)  # 返回张量的第一个维度
 
 
class SkipGram(nn.Module):
    def __init__(self, embedding_size):
        super(SkipGram, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = torch.nn.Linear(vocab_size, self.embedding_size)
        self.fc2 = torch.nn.Linear(self.embedding_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()
 
    def forward(self, center, context):
        """
        :param center: [Batch_size]
        :param context:[Batch_size, vocab_size]
        :return:
        """
        center = self.fc1(center)
        center = self.fc2(center)
        loss = self.loss(center, context)
        return loss
 
 
batch_size = 2
center_data, context_data = make_data(sentences)
train_data = my_dataset(center_data, context_data)
train_iter = DataLoader(train_data, batch_size, shuffle=True)
epochs = 2000
model = SkipGram(embedding_size=embedding_size)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(epochs):
    for center, context in train_iter:
        loss = model(center, context)
        if epoch % 100 == 0:
            print("step {0} loss {1}".format(epoch, loss.detach().numpy()))
        optim.zero_grad()
        loss.backward()
        optim.step()
