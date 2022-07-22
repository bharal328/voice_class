import re
import csv
from config import config
import pandas as pd
import jieba
import jieba.analyse
import numpy as np
import random


#用于在一个整数范围内产生n个随机数，注意范围内的整数个数要小于所需的随机数的个数
#第一个参数表示开始的范围，第二个参数表示结束的范围，第三个表示产生随机数的个数
#最后返回的是一个列表

class CreateRandomPage:
    def __init__(self, begin, end, needcount):
        self.begin = begin
        self.end = end
        self.needcount = needcount
        self.resultlist = []
        self.count = 0
    def createrandompage(self):
        tempInt = random.randint(self.begin, self.end)
        if(self.count < self.needcount):
            if(tempInt not in self.resultlist):
                self.resultlist.append(tempInt)    #将长生的随机数追加到列表中
                self.count += 1
            return self.createrandompage()      #在此用递归的思想
        return self.resultlist

class extract_words_one_file():
    def __init__(self):
        jieba.load_userdict(config["userdic"])

    #只保留中文字符
    def remove(self, text):
        remove_chars = r'[^\u4e00-\u9fd5]'
        return re.sub(remove_chars, '', text)

    #打开stopwords文件函数声明
    def open_stop(self, file_stop):
        stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]
        return stopwords

    #利用jieba进行分词
    def seg_sentence(self, sentence):

        sentence_seged = jieba.posseg.cut(sentence.strip())
        stopwords = self.open_stop(config["stopwords"])
        outstr = ''
        part = []
        for word, flag in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
                    part.append(flag)
        return outstr.strip(), part

dic = {"memory": [], "understand": [], "application": [], "analysis": []}
dic_key = list(dic.keys())
alltokens = [[] for i in range(len(dic_key))]  # 储存每个类中一句话的分词

def DataPrepare(file):
    Extract = extract_words_one_file()

    sen = {"memory": [], "understand": [], "application": [], "analysis": []}
    wordpart = {"memory": [], "understand": [], "application": [], "analysis": []}  # 储存词性

    alltokensWithLabel = []  # [alltokens, label]

    train = []  # 训练集
    test = []  # 测试集，3份数据里2份训练1份测试
    y_train = []
    y_test = []

    train_wordPart = []
    test_wordPart = []


    with open(file,'r',encoding='utf-8') as csvfile:
        reader = pd.read_csv(csvfile)

        for i in range(4):
            for s in reader[dic_key[i]][1:]:
                if s == s:
                    sen[dic_key[i]].append(s)
            # print(len(sen[dic_key[i]]))

        memoryRand = CreateRandomPage(0, len(sen["memory"])-1, len(sen["memory"])*0.25)
        understandRand = CreateRandomPage(0, len(sen["understand"])-1, len(sen["understand"])*0.25)
        applicationRand = CreateRandomPage(0, len(sen["application"])-1, len(sen["application"])*0.25)
        analysisRand = CreateRandomPage(0, len(sen["analysis"])-1, len(sen["analysis"])*0.25)
        all_rand = [memoryRand, understandRand, applicationRand, analysisRand]

        with open('train_seg.txt', 'w') as f:
            with open("train.txt", 'w', encoding='utf-8') as t:
                for id, rand in enumerate(all_rand):
                    temp = set()
                    for i in rand.createrandompage():
                        temp.add(i)
                        sentence = sen[dic_key[id]][i]
                        remove_sentence = Extract.remove(sentence)
                        line_seg, part = Extract.seg_sentence(remove_sentence)  # 这里的返回值是字符串
                        wordpart[dic_key[id]].append(part)
                        word = re.split(r'\s', line_seg)

                        test.append(word)
                        y_test.append(id)
                        alltokensWithLabel.append([word, id])
                    rest = set(range(len(sen[dic_key[id]]))) - temp

                    for i in rest:
                        sentence = sen[dic_key[id]][i]
                        remove_sentence = Extract.remove(sentence)

                        line_seg, part = Extract.seg_sentence(remove_sentence)  # 这里的返回值是字符串

                        f.write(line_seg)
                        f.write('\n')

                        wordpart[dic_key[id]].append(part)
                        word = re.split(r'\s', line_seg)


                        train.append(word)
                        y_train.append(id)

                        t.write(' '.join(word))
                        t.write('\t' + str(id) + '\n')

                        alltokensWithLabel.append([word, id])

            feature = pd.DataFrame(alltokensWithLabel, columns=['Words', 'Label'])  # 将分词、种类以dataframe储存
            feature.to_csv(config["segData"], encoding='utf_8_sig', index=False)

            # print("train", train)
            print("train: memory:{}, understand:{}, application:{}, analysis:{}".format(y_train.count(0), y_train.count(1), y_train.count(2),  y_train.count(3)))
            # print("test", test)
            print("test: memory:{}, understand:{}, application:{}, analysis:{}".format(y_test.count(0), y_test.count(1), y_test.count(2), y_test.count(3)))
            # print("train:{}, test:{}".format(len(train), len(test)))

            f.close()
            return train, y_train, test, y_test, train_wordPart, test_wordPart

def wordTokenTrain():
    import gensim
    # print(config["pre_tokenTxt"])

    train, y_train, test, y_test, train_wordPart, test_wordPart = DataPrepare(config["srcData"])

    # 导入数据
    # 首先初始化一个word2vec 模型：
    w2v_model = gensim.models.Word2Vec(size=300, window=5, min_count=2)
    w2v_model.build_vocab(train)
    # 再加载第三方预训练模型：

    third_model = gensim.models.KeyedVectors.load_word2vec_format(config["pre_tokenTxt"], binary=False)
    # 通过 intersect_word2vec_format()方法merge词向量：
    w2v_model.build_vocab([list(third_model.vocab.keys())], update=True)
    w2v_model.intersect_word2vec_format(config["pre_tokenTxt"], binary=False, lockf=1.0)
    w2v_model.train(train, total_examples=w2v_model.corpus_count, epochs=10)
    # print("Model training finished.")

    w2v_model.wv.save_word2vec_format(config["afterTrainToken"], binary=False)
    # print("Model saved.")
    # print(config["afterTrainToken"])

    return train, y_train, test, y_test, train_wordPart, test_wordPart


if __name__ == '__main__':
    np.random.seed(44)
    random.seed(44)
    # wordTokenTrain()
    DataPrepare(config["srcData"])
