import numpy as np


# 将相似度矩阵代入softmax公式，可以得到一个满足概率分布的矩阵
def softmax(x):
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出
        x -= tmp.reshape((x.shape[0], 1))  # 使每行减去所在行的最大值

        x = np.exp(x)  # 第一步，计算所有值以e为底的x次幂
        tmp = np.sum(x, axis=1)  # 将每行求和并保存
        x /= tmp.reshape((x.shape[0], 1))  # 所有元素除以所在行的元素和（广播运算）

    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求以e为底的x次幂
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x


# 计算softmax的梯度和代价，假设给定中心词要预测'c'，预测下一个字母是'd'
def softmaxCostAndGradient(predicted, target, outputVectors):
    '''
    predicted：输入词向量，也就是例子中'c'的词向量
    target：目标词向量的索引，也就是真实值'd'的索引
    outputVectors：输出向量矩阵
    '''
    v_hat = predicted  # 中心词向量
    z = np.dot(outputVectors, v_hat)  # 输出矩阵和中心词向量的内积，预测得分
    y_hat = softmax(z)  # 预测输出y_hat
    cost = -np.log(y_hat[target])  # 计算代价
    z = y_hat.copy()
    z[target] -= 1.0
    grad = np.outer(z, v_hat)  # 计算中心词的梯度
    gradPred = np.dot(outputVectors.T, z)  # 计算输出词向量矩阵的梯度
    return cost, gradPred, grad

# skip-gram算法
def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors):
    '''
    currentWord:当前的中心词
    contextWords:待预测的词
    tokens:用于映射字母在输入输出矩阵中的索引
    inputVectors: 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
    outputVectors：输出矩阵
    '''
    # 初始化变量
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    cword_idx = tokens[currentWord]  # 得到中心单词的索引
    v_hat = inputVectors[cword_idx]  # 得到中心单词的词向量

    # 循环预测上下文中每个字母
    for j in contextWords:
        u_idx = tokens[j]  # 得到目标字母的索引
        c_cost, c_grad_in, c_grad_out = softmaxCostAndGradient(v_hat, u_idx, outputVectors)  # 计算一个中心字母预测一个上下文字母的情况
        cost += c_cost  # 所有代价求和
        gradIn[cword_idx] += c_grad_in  # 中心词向量梯度求和
        gradOut += c_grad_out  # 输出词向量矩阵梯度求和
    return cost, gradIn, gradOut


inputVectors = np.random.randn(5, 3)  # 输入矩阵，语料库中字母的数量是5，我们使用3维向量表示一个字母
outputVectors = np.random.randn(5, 3)  # 输出矩阵
sentence = ['a', 'e', 'd', 'b', 'd', 'c', 'd', 'e', 'e', 'c', 'a']  # 句子
centerword = 'c'  # 中心字母
context = ['a', 'e', 'd', 'b', 'd', 'd', 'e', 'e', 'c', 'a']  # 上下文字母
tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])  # 用于映射字母在输入输出矩阵中的索引
c, gin, gout = skipgram(centerword, context, tokens, inputVectors, outputVectors)
step = 0.01  # 更新步进
print('原始输入矩阵:\n',inputVectors)
print('原始输出矩阵:\n',outputVectors)
inputVectors -= step * gin  # 更新输入词向量矩阵
outputVectors -= step * gout
print('更新后的输入矩阵:\n', inputVectors)
print('更新后的输出矩阵:\n', outputVectors)
