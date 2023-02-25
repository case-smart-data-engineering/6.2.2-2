#!/usr/bin/env python3

# 待测试程序
def solution(words):
    '''
    1 获取文本语料
    '''
    with open('1_算法示例/news.txt', 'r', encoding='utf-8') as f:
        news = f.readlines()

    stop_list = set('for a of the and to in <unk>'.split(' '))
    # 对每篇文档进行转小写，并且以空格分割，同时过滤掉所有的停止词
    texts = [[word for word in document.lower().split() if word not in stop_list] for document in news]
    '''
    2 载入数据，训练并保存模型
    '''
    from gensim.models import word2vec

    sentences = word2vec.Text8Corpus('1_算法示例/news.txt')  # 将语料保存在sentence中
    model = word2vec.Word2Vec(sentences, sg=1, vector_size=100, window=5, min_count=5, negative=3, sample=0.001, hs=1,
                            workers=4)  # 生成词向量空间模型
    '''
    sg:用于设置训练算法,默认为0,对应CBOW算法;sg=1则采用skip-gram算法
    vector_size:是指输出的词的向量维数,默认为100,大的size需要更多的训练数据,但是效果会更好,推荐值为几十到几百
    window:为训练的窗口大小,8表示每个词考虑前8个词与后8个词,默认为5
    min_count: 可以对词库做过滤,词频少于min_count次数的单词会被丢弃掉, 默认值为5
    negative: 用于设置多少个负采样个数
    hs: word2vec两个解法的选择
    workers: 训练的并行个数
    '''
    model.save('3_自测练习/text_word2vec.model')  # 保存模型

    '''
    3 加载模型，实现各个功能
    '''
    # 加载模型
    model = word2vec.Word2Vec.load('3_自测练习/text_word2vec.model')
    # 寻找不合群的词
    result = model.wv.doesnt_match(words.split())

if __name__ == '__main__':
    pass
