import jieba
from gensim.models import Word2Vec
sentence="今天天气真的很好！"
def seg_sentence(sentence):
    """"进行分词"""
    sentence_seged = jieba.cut(sentence.strip())  #分词
    stopwords = [' ']
    outstr = ''   # 必须字符，不能列表
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    # return outstr
    return outstr.split(' ')  # 以空格分割 列表
def vec_produce(sentence,word,size):
    """生成词向量"""
    sentenceseg = seg_sentence(sentence) # 已分词可向量化的句子
    model = Word2Vec(sentences=[sentenceseg], vector_size=size, window=5, min_count=1, workers=4)
    word_vectors = model.wv
    wordvec = word_vectors[word]
    print(wordvec)
    return wordvec

vec_produce(sentence,seg_sentence(sentence),128)