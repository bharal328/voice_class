import numpy as np
from config import config
from dataprepare import DataPrepare, wordTokenTrain, extract_words_one_file
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer  # 抽取特征：TF-IDF
# from sklearn.metrics import classification_report
# from sklearn import svm, naive_bayes
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import  AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# import math
import gensim
import random
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score

np.set_printoptions(threshold=np.inf)

# w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('./w2v.6B.300d.txt', binary=False)
w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(config["afterTrainToken"], binary=False)
tfidf_w2v={}
test_tfidf_w2v = {}

all_vocab = w2vmodel.vocab.keys()
label = ["记忆", "理解", "应用", "分析"]

#num_features表示的文本单词大小
def average_word_vectors(words,model,num_features, vocab=None, fuseFlag=False):
    feature_vector=np.zeros((num_features,),dtype='float64')
    nwords=0
    if fuseFlag == True:
        for word in words:
            if word in model.keys():
                nwords=nwords+1

                feature_vector=np.add(feature_vector,model[word])
        if nwords:
            feature_vector=np.divide(feature_vector,nwords)
        return feature_vector

    else:
        for word in words:
            if word in vocab:
                nwords = nwords + 1
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector=np.divide(feature_vector,nwords)
        return feature_vector.tolist()

def averaged_word_vectorizer(corpus, model, num_features, fuseFlag, trainFlag=True):
    #get the all vocabulary
    if fuseFlag:
        if trainFlag:
            features = [average_word_vectors(tokenized_sentence, model, num_features, fuseFlag=fuseFlag) for tokenized_sentence in corpus]
            return np.array(features)
        else:
            features = [average_word_vectors(corpus, model, num_features, fuseFlag=fuseFlag)]
            return np.array(features)
    else:
        vocabulary = set(model.index2word)
        features = [average_word_vectors(tokenized_sentence, model, num_features, vocabulary, fuseFlag=fuseFlag) for tokenized_sentence in corpus]
        return np.array(features)

def get_word_vectors(data, fuseFlag=False, trainFlag=True):
    if fuseFlag:
        if trainFlag:
            return averaged_word_vectorizer(data, model=tfidf_w2v, num_features=300, fuseFlag=fuseFlag, trainFlag=trainFlag)
        else:
            return averaged_word_vectorizer(data, model=test_tfidf_w2v, num_features=300, fuseFlag=fuseFlag, trainFlag=trainFlag)
    else:
        # w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('w2v.6B.300d.txt', binary=False)
        w2vmodel = gensim.models.KeyedVectors.load_word2vec_format(config["afterTrainToken"], binary=False)
        return averaged_word_vectorizer(data, model=w2vmodel, num_features=300, fuseFlag=fuseFlag)

# def confusion(y_pred, y_test, test, flag, printFalse=False):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix
#     import numpy as np
#
#     if printFalse:
#         for i in range(len(y_test)):
#             if y_pred[i] != y_test[i]:
#                 print(test[i], "y_pred:{}, y_test:{}".format(y_pred[i]+1, y_test[i]+1))
#
#     # 绘制confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     # print("Confusion Matrix")
#
#     dic = {"memory": [], "understand": [], "application": [], "analysis": []}
#     category_labels = list(dic.keys())
#
#     cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     # print(cm_normalised)
#     sns.set(font_scale=1.5)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax = sns.heatmap(cm_normalised, annot=True, linewidths=1, square=False,
#                      cmap="Reds", yticklabels=category_labels, xticklabels=category_labels, vmin=0,
#                      vmax=np.max(cm_normalised),
#                      fmt=".2f", annot_kws={"size": 20})
#     ax.set(xlabel='Predicted label', ylabel='True label')
#     if flag:
#         plt.show()

def test(seed, findseed=False, sentence=""):
    np.random.seed(seed)
    random.seed(seed)
    import pickle
    import joblib
    import re
    Extract = extract_words_one_file()  ##text

    # clf = pickle.load("dtr.dat")
    f = open('myModel.model', 'rb')  # 注意此处model是rb
    s = f.read()
    clf = pickle.loads(s)

    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    #sentence = input("请输入:")

    remove_sentence = Extract.remove(sentence)
    line_seg, part = Extract.seg_sentence(remove_sentence)  # 这里的返回值是字符串
    test = re.split(r'\s', line_seg)
    all_testWord = []
    temp = ""
    for word in test:
        temp += word
        temp += " "
    all_testWord.append(temp[:-1])
    X_test_tfidf = tfidf_vectorizer.transform(all_testWord)
    # print(X_test_tfidf)

    test_feature_names = tfidf_vectorizer.get_feature_names_out()
    test_shape = X_test_tfidf.shape[0]

    for i in range(test_shape):
        for index, j in enumerate(X_test_tfidf[i][0].indices):
            word = test_feature_names[j]
            if word in all_vocab:
                test_tfidf_w2v[word] = w2vmodel[word] * X_test_tfidf[i, j]

    tfidf_w2vTest = get_word_vectors(test, True, False)
    # print(tfidf_w2vTest)
    predicted_svm = clf.predict(tfidf_w2vTest)

    print("认知等级为:", label[predicted_svm[0]])
    return label[predicted_svm[0]]

def findSeed(t1, t2, w2vFlag, fuseFlag, posTfidf, findseed):
    from tqdm import tqdm

    ans = {}
    with tqdm(total=t2-t1) as t:
        try:
            for seed in range(t1, t2):
                x = test(seed=seed, w2vFlag=w2vFlag, fuseFlag=fuseFlag, posTfidf=posTfidf, findseed=findseed)
                if x:
                    ans[str(seed)] = x
                t.update(1)
            print(ans)
        except:
            print(ans)
        finally:
            ans = sorted(ans.items(), key=lambda x: x['accuracy'])
            print(ans)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # w2v and not fuseFlag and not postfidf:            w2v
    # not w2vFlag and not fuseFlag and not postfidf:    tfidf
    # postfidf and not fuseFlag and not w2v:            postfidf
    # w2vFlag and fuseFlag and not postfidf:            w2v + tfidf
    # fuseFlag and posTfidf and w2vFlag:                w2v + posTfidf

    Extract = extract_words_one_file()
    test(44, trainFlag=False,sentence=" 嗯共创原告")
