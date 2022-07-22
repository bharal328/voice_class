import argparse
import functools
import os
import shutil
import numpy as np
import torch
from utils.reader import load_audio,MFCC
from utils.utility import add_arguments, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('input_shape',      str,    '(1, 257, 257)',          '数据输入的形状')
add_arg('threshold',        float,   0.70,                    '判断是否为同一个人的阈值')
add_arg('audio_db',         str,    'audio_init_Library',               '音频库的路径')
add_arg('model_path',       str,    'models/resnet34.pth',    '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

device = torch.device("cuda")

model = torch.jit.load(args.model_path)
model.to(device)
model.eval()

person_feature = []
person_name = []


def infer(audio_path):
    input_shape = eval(args.input_shape)
    #data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    data = MFCC(audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    feature = model(data)
    return feature.data.cpu().numpy()


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]  #输出每个音频对应的特征值

        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


def recognition(path):
    name = ''
    pro = 0
    dis={}
    all_feature=infer(path)
    #print('all_feature',all_feature)
    feature = all_feature[0]
    for i, person_f in enumerate(person_feature):

        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))##求特征的对角余弦值   矩阵相乘除以矩阵范数乘积
        dis[person_name[i]]=dist
        if dist > pro:
            pro = dist
            name = person_name[i]
    print('准确度排名：',sorted(dis.items(),key=lambda item:item[1],reverse=True))
    return name, pro    #返回名字和最高的准确度


# 声纹注册
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])  #[-4:]是.wav
    shutil.move(path, save_path)
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)


if __name__ == '__main__':
    from system import RecordAudio
    load_audio_db(args.audio_db)
    record_audio = RecordAudio()
    #audio = process_signal()
    while True:
        select_fun = input("请选择功能，0为注册音频到声纹库，1为执行声纹识别")
        if select_fun == "0":
            audio_path = record_audio.record()
            name = input("请输入该音频用户的名称：")
            if name == '': continue
            register(audio_path, name)


        elif select_fun == "1":
            audio_path = record_audio.record()
            name, p = recognition(audio_path)
            if p > args.threshold:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
                #record_audio.play(audio_path)

            else:
                #record_audio.play(audio_path)
                print("音频库没有该用户的语音")
        else:
            print('请正确选择功能')

