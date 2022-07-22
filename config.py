config = {
    "stopwords":   r"D:\DL\VoiceprintRecognition-Pytorch-master\VoiceprintRecognition-Pytorch-master/stopwords.txt",                   # 这里面存要去掉的词
    "userdic":   r"D:\DL\VoiceprintRecognition-Pytorch-master\VoiceprintRecognition-Pytorch-master/userdic.txt",                       # jieba分词，这里面存不希望被分开的词
    "afterTrainToken":  r"D:\DL\VoiceprintRecognition-Pytorch-master\VoiceprintRecognition-Pytorch-master/word2vec_ensemble(综合语料库15ep).txt",       # 训练后的词向量模型
    # "afterTrainToken": "./bert_vectors.txt",
    "vif": 2,  # 动词权重
    "aif": 1, # 形容词、名词权重
    "nif": 1
}
