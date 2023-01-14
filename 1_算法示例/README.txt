1.py为word2vec的pytorch实现
这里有两个加速算法，一个是negative sampling 一个是softmax 

由于整个模型的输出是一个softmax全连接，全连接的维度为词汇表 的长度

由于实际使用时词汇表长度可能过大，做softmax归一化时间太长，并且我们训练Word2Vec模型的主要目的不是想要预测下一个词语，而是得到一个Embedding矩阵对token进行封装。所以这里我们可以在构造正样本的同时，构造更多数量的负样本

eg: I like china very much  模型的window_size 为2 

那么单次china 构造的正样本对为{china , like } , {china , very}  

负样本对为{china , I }  {china , much }   通过构造负样本可以让模型更快的收敛



word2vec.py为利用gensim中的word2vec函数生成词向量
示例数据为sentence="今天天气真的很好！"可以自己定义也可以读取文本获得。SIZE为向量维度这里使用的是128