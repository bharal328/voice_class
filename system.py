from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QIcon
from infer_recognition import *
import voice
import sys
from pytexts import *
import websocket
import wave
import  shutil
# from dataprepare import extract_words_one_file
from SVM import test
import pyaudio
import librosa
from re import findall
app = QApplication(sys.argv)
ui = voice.Ui_MainWindow()      ##放在这里是因为下面函数要用


audio_path=None
register_mode=False

student_level={}
speak_times={}
ans=""
pre_w=""
pre_end_w=""
pre_len=0

class Analysis_thread(QThread):
    sig=pyqtSignal(str)
    def __init__(self):
        super(Analysis_thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        """
        进行任务操作，主要的逻辑操作,返回结果
        """
        print("开始认知分析！！！")
        res=level_analysis()
        self.sig.emit(str(res))


def callback(msg):
    name,acc=msg.split()
    QtWidgets.QMessageBox.information(QWidget(), "提示", "识别为："+name+"准确率："+acc)

class Record_Thread(QThread):
    signal = pyqtSignal(str)  # 括号里填写信号传递的参数

    def __init__(self):
        super(Record_Thread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        """
        进行任务操作，主要的逻辑操作,返回结果
        """
        #print(111)
        print("开始录音")
        msg=record()
        self.signal.emit(msg)  # 发射信号


class MyWindows(QtWidgets.QMainWindow,voice.Ui_MainWindow):
    def __init__(self):
        super(MyWindows, self).__init__()
        self.setupUi(self)


    # 点击事件启动线程
    def buttonClick(self):
        self.thread = Record_Thread()
        #self.web_thread=web_Thread()
        #self.thread.signal.connect(self.callback)  # 连接回调函数，接收结果
        self.thread.signal.connect(callback)
        self.thread.start()  # 启动线程





def class_record():
    global ws
    ui.textBrowser.setText("讲话完毕后可按停止录音")
    windows.buttonClick()
    # text_thread=Text_Thread()
    # text_thread.start()





def record():
    global ans,sentence
    print("push botton3")
    try :
        audio_path = record_audio.record(class_record=True)
    except Exception as e:
        print('错误明细是',e.__class__.__name__,e)
    print("audio_path",audio_path)
    name, p = recognition(audio_path)
    if p > args.threshold:
        print("识别说话的为：%s，相似度为：%f" % (name, p))
        #ui.textBrowser.setText("识别说话的为：%s，相似度为：%f" % (name, p))
    else:
        print("音频库没有该用户的语音")
        #ui.textBrowser.setText("音频库没有该用户的语音")

    if name not in speak_times:
        speak_times[name]=1
    else:
        speak_times[name]+=1
    save_path=os.path.join("class_audio",name+str(speak_times[name])+'.WAV')
    shutil.move(audio_path,save_path)
    print("已将",audio_path,"转移到",save_path)
    return name+' '+str(p)


class RecordAudio:
    def __init__(self):
        # 录音参数
        global p
        self.chunk = 1024  #帧数
        self.format = pyaudio.paInt16
        self.channels = 1   #声道数
        self.rate = 16000   #采样率
        #self.pause_flag=False
        # 打开录音
        p = pyaudio.PyAudio()
        self.stream = p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def record(self, output_path="temp_audio/temp.wav", record_seconds=8,class_record=False):
        """
        录音
        :param output_path: 录音保存的路径，后缀名为wav
        :param record_seconds: 录音时间，默认3秒
        :return: 录音的文件路径
        """
        global pause_flag
        from tqdm import tqdm

        frames = []
        if not class_record:
            print("注册录音中")
            for i in tqdm(range(0, int(self.rate / self.chunk * record_seconds))):
                data = self.stream.read(self.chunk)
                frames.append(data)
        else:
            # print("录音中")
            while True:
                pause_flag=ui.pushButton_4.isDown()
                if pause_flag == False:
                    data = self.stream.read(self.chunk)
                    frames.append(data)
                else:
                    break

        print("录音已结束!")
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return output_path

    def play(self):
        print("正在转文本")
        files=os.listdir("class_audio")
        path=os.path.join("class_audio",files[0])
        print("path",path)
        wf=wave.open(path,'rb')#rb为以二进制格式打开一个文件只读   字符串前面有个r 是防止字符转义
        self.stream = p.open(format=self.format,
                                  channels=1,
                                  rate=16000,
                                  output=True,       #输出格式
                                  frames_per_buffer=self.chunk)
        data=wf.readframes(self.chunk)
        # while data !=b'':
        #
        #     #print("data",data)
        #     # stream.write(data)
        #     # data=wf.readframes(self.chunk)
        #     #print('data',data)
        self.stream.stop_stream()
        self.stream.close()
        print('播放结束！')


def register_man():#按注册录音
    global register_mode
    ui.lineEdit_2.setPlaceholderText("注册时在此框输入名字")
    ui.textBrowser.setText("先在旁边框中输入用户名称。再按下“开始注册录音”键录音，自动录音8秒钟")
    register_mode=True
    #class_record_mode=False#不进入课堂录音模式

def register_record():#按开始录音
    global register_mode
    if register_mode:
        #register_flag=False
        name = ui.lineEdit_2.text()
        if name == '':
            QtWidgets.QMessageBox.warning(QWidget(), "警告", "名字为空，请重新输入名字！")
            ui.textBrowser.setText("名字为空，请重新输入名字！")
            return
        else:
            try:
                path = record_audio.record()
            except Exception as e:
                print('注册错误明细是',e.__class__.__name__,e)
            ui.textBrowser.setText("注册录音已结束！")
            register(path, name)
    QtWidgets.QMessageBox.information(QWidget(), "提示", "注册已完成！")

def analysis_thread():
    try:
        print("开始线程！！！")
        thread=Analysis_thread()
        thread.sig.connect(an_callback)
        thread.start()
    except Exception as e :
        print('分析错误明细是', e.__class__.__name__, e)

def an_callback(msg):
    ui.textBrowser.setText("学生认知水平等级分析为：" + msg)
def level_analysis():
    files=os.listdir("./class_audio")
    for file in files:
        name = findall("[\u4e00-\u9fa5]", file)
        name = "".join(name)
        #print(name)
        print("name",name)
        print("filename", file)
        try:
            api = RequestApi(appid="736d11ff", secret_key="ccd8f25fbaf073b91f89013f25dfec8e",
                         upload_file_path=os.path.join(os.getcwd(),"class_audio",file))
            analysis_ans = api.all_api_request()
        except Exception as e:
            print('认知等级分析错误明细是',e.__class__.__name__,e)
        print("analysis_ans",analysis_ans)


        try:
            level=test(44, sentence=analysis_ans)
            if name not in student_level.keys():
                student_level[name]=[level]
            else:
                student_level[name].append(level)
            print("student_level",student_level)
        except Exception as e:
            print('错误明细是',e.__class__.__name__,e)
        print('-' * 70)
    return student_level

if __name__=="__main__":
    global record_audio
    record_audio = RecordAudio()
    websocket.enableTrace(False)

    windows = MyWindows()
    ui.setupUi(windows)

    load_audio_db(args.audio_db)
    windows.setWindowIcon(QIcon("img/pic.png"))  ##一定得放在show()前面，咱也不知道为啥
    windows.show()
    print("加载完成")
    windows.setWindowTitle("课堂教学反馈系统")
    ui.pushButton_3.setToolTip("停止时按停止录音")
    ui.textBrowser.setText('注册声纹库：'+ ' '.join(person_name))
    ui.pushButton.clicked.connect(register_man)         ##注册模式
    ui.pushButton_2.clicked.connect(register_record)    ##注册录音
    ui.pushButton_3.clicked.connect(class_record)       ##课堂录音
    ui.pushButton_5.clicked.connect(analysis_thread)


    sys.exit(app.exec_())





