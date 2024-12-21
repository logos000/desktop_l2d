from ctypes import windll
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QTabWidget, QGroupBox, QFormLayout, QLineEdit, QFileDialog, QHBoxLayout, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QTextCursor
import numpy as np
import sounddevice as sd
import wave
import tempfile
import openai
from dotenv import load_dotenv
import json
import io
import requests
import soundfile as sf
from funasr import AutoModel
import time
import pyaudio
import threading
from queue import Queue
import uuid
from openai import OpenAI
import logging
import asyncio
import qasync

from vts_connector.vts_manager import VTSManager
from vts_connector.vts_manager_async import VTSManagerAsync
from tts_service.tts_client import TTSClient
from gpt_service.gpt_client import GPTClient
from gpt_service.tools.screenshot import screenshot_tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

class AsyncHelper:
    """异步操作助手类"""
    def __init__(self):
        self._loop = None
        
    def set_loop(self, loop):
        """设置事件循环"""
        self._loop = loop
        
    async def _run_async(self, coro):
        """运行异步协程"""
        try:
            return await coro
        except Exception as e:
            print(f"异步操作错误: {str(e)}")
            return None
            
    def run_async(self, coro):
        """在事件循环中运行异步协程"""
        if self._loop is None:
            print("错误: 事件循环未初始化")
            return None
        return asyncio.run_coroutine_threadsafe(self._run_async(coro), self._loop)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 加载环境变量
_ = load_dotenv(override=True)

# 初始化OpenAI客户端
openai_client = openai.OpenAI()

# 从配置中读取实时识别设置
enable_realtime_asr = os.getenv('ENABLE_REALTIME_ASR', 'True').lower() == 'true'

# 初始化语音识别模型
print("正在加载语音识别模型...")
# 只有在启用实时识别时才加载流式识别模型
streaming_asr_model = None
if enable_realtime_asr:
    print("加载流式识别模型...")
    streaming_asr_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", device="cuda:0", disable_progress_bar=True, disable_update=True)
    # 预热模型
    print("预热流式识别模型...")
    test_audio = np.zeros(800, dtype=np.float32)
    streaming_asr_model.generate(
        input=test_audio,
        cache={},
        is_final=False,
        chunk_size=[0, 10, 5],
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1
    )

# 设置模型路径
MODEL_DIR = os.path.join(os.getcwd(), "models", "SenseVoiceSmall")

# 确保模型目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

# 检查模型是否已下载
if not os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
    print("首次运行，正在下载模型...")
    # 下载模型到指定目录
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='iic/SenseVoiceSmall',
        model_revision="master",
        device="cuda:0",
        model_kwargs={'lang': 'zh'}
    )
    # 复制模型文件到本地目录
    import shutil
    model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "iic", "SenseVoiceSmall")
    if os.path.exists(model_cache_dir):
        print(f"正在复制模型文件到 {MODEL_DIR}")
        shutil.copytree(model_cache_dir, MODEL_DIR, dirs_exist_ok=True)
        print("模型文件复制完成")
else:
    print("使用本")

# 使用本地模型
non_streaming_asr_model = pipeline(
    task=Tasks.auto_speech_recognition,
    model=MODEL_DIR,
    device="cuda:0",
    model_kwargs={'lang': 'zh'},
    disable_update=True
)

def clean_asr_result(result):
    """清理ASR结果"""
    print(f"ASR Result: {result}")
    print(f"ASR Result Type: {type(result)}")
    
    try:
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], dict):
                text = result[0].get('text', '')
                print(f"Extracted Text: {text}")
                return text.strip()
        elif isinstance(result, dict):
            text = result.get('text', '')
            print(f"Extracted Text: {text}")
            return text.strip()
        elif isinstance(result, str):
            print(f"Text String: {result}")
            return result.strip()
        else:
            print(f"Unknown Result Type: {type(result)}")
            return ""
    except Exception as e:
        print(f"Error cleaning ASR result: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

class ResponseHandler(QObject):
    """独立的响应处理服务"""
    response_received = Signal(str)  # 完整响应信号
    sentence_received = Signal(str)  # 完整句子信号
    partial_response = Signal(str)  # 部分响应信号
    response_completed = Signal(str)  # 响应完成信号，用于更新消息历史
    
    def __init__(self):
        super().__init__()
        self.response_queue = Queue()  # 响应队列
        self.is_running = True
        self.process_thread = None
        
        # 初始化GPT客户端
        self.gpt_client = GPTClient()
        
        # 创建流式输出回调处理器
        class StreamingCallbackHandler(BaseCallbackHandler):
            def __init__(self, response_handler):
                self.response_handler = response_handler
                
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                if token:
                    self.response_handler.add_response(token)
        
        # 设置回调处理器
        self.streaming_handler = StreamingCallbackHandler(self)
        
        # 创建测试执行器
        self.agent_executor = self.gpt_client.create_test_executor(self.streaming_handler)
    
    def start(self):
        """启动处理服务"""
        if self.process_thread is None or not self.process_thread.is_alive():
            self.is_running = True
            self.process_thread = threading.Thread(target=self._process_thread)
            self.process_thread.daemon = True
            self.process_thread.start()
            
    def stop(self):
        """停止处理服务"""
        self.is_running = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join()
        # 清空队列
        while not self.response_queue.empty():
            self.response_queue.get()
            
    def add_response(self, content):
        """添加响应内容到队列"""
        if content is not None:
            self.response_queue.put(content)
            
    def _process_thread(self):
        """响应处理线程"""
        current_response = ""
        current_sentence = ""
        
        while self.is_running:
            try:
                if not self.response_queue.empty():
                    content = self.response_queue.get()
                    if content:
                        # 更新当前响应
                        current_response += content
                        self.partial_response.emit(current_response)
                        
                        # 更新当前句子
                        current_sentence += content
                        if any(p in content for p in ['。', '！', '？', '.', '!', '?', '\n']):
                            if current_sentence.strip():
                                self.sentence_received.emit(current_sentence.strip())
                                current_sentence = ""
                                
            except Exception as e:
                print(f"处理响应时出错: {str(e)}")
                import traceback
                traceback.print_exc()
            time.sleep(0.01)  # 避免CPU占用过高
            
    def process_message(self, text: str):
        """处理消息并获取响应"""
        try:
            # 构建Agent消息
            agent_message = {
                "input": text,
                "chat_history": self.gpt_client.chat_history
            }
            
            # 使用Agent执行
            response = self.agent_executor.invoke(agent_message)
            
            # 更新对话历史
            self.gpt_client.chat_history.extend([
                HumanMessage(content=text),
                AIMessage(content=response["output"])
            ])
            
            # 发送完整响应
            if response and "output" in response:
                self.response_completed.emit(response["output"])
                
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            print(error_msg)
            self.response_completed.emit(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()

class AudioRecorder(QObject):
    # 信号定义
    volume_update = Signal(float)  # 音量更新
    correction_result = Signal(str)  # 纠错结果信号
    partial_result = Signal(str)  # 实时识别结果信号

    def __init__(self):
        super().__init__()
        print("正在加载语音识别模型...")
        # 流式识别模型
        self.streaming_model = streaming_asr_model  # 使用全局变量
        # 非识别模型
        self.offline_model = AutoModel(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", disable_update=True)
        
        # 音频参数
        self.sample_rate = 16000
        self.chunk_size = 3200  # 使用与播放相同的块大小
        
        # 语音检测参数
        self.ENERGY_THRESHOLD = 0.005
        self.VOLUME_THRESHOLD = 0.01
        self.SILENCE_CHUNKS = 20
        self.ACCUMULATION_CHUNKS = 10
        
        # 缓冲区
        self.audio_buffer = []
        self.text_buffer = []
        
        # 状态标志
        self.is_recording = False
        self.is_final = False
        
        # 音频队列
        self.audio_queue = Queue(maxsize=50)
        
        # 音频流
        self.stream = None
        
        # 实时识别设置
        self.enable_realtime_asr = enable_realtime_asr  # 使用全局变量
        
        print("语音识别模型加载完成")

    def set_realtime_asr(self, enable: bool):
        """设置是否启用实时识别"""
        global streaming_asr_model
        
        # 如果状态没有改变，直接返回
        if self.enable_realtime_asr == enable:
            return
            
        self.enable_realtime_asr = enable
        
        # 如果启用实时识别，但模型未加载，则加载模型
        if enable and streaming_asr_model is None:
            print("加载流式识别模型...")
            streaming_asr_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", device="cuda:0", disable_progress_bar=True, disable_update=True)
            # 预热模型
            print("预热流式识别模型...")
            test_audio = np.zeros(800, dtype=np.float32)
            streaming_asr_model.generate(
                input=test_audio,
                cache={},
                is_final=False,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            self.streaming_model = streaming_asr_model
        elif not enable:
            # 如果禁用实时识别，释放模型
            streaming_asr_model = None
            self.streaming_model = None

    def _audio_callback(self, indata, frames, time, status):
        """音频回调处理"""
        if status:
            print(f"录音状态: {status}")
            return
        
        try:
            # 将数据转换为float32类型
            audio_chunk = indata.flatten().astype(np.float32)
            
            # 计算音量并发送信号
            volume = float(np.sqrt(np.mean(np.square(audio_chunk))))
            self.volume_update.emit(volume)
            
            # 如果音量低于阈值，将音频数据置零
            if volume < self.VOLUME_THRESHOLD:
                audio_chunk = np.zeros_like(audio_chunk)
            
            # 使用 try 避免队列满时的阻塞
            try:
                self.audio_queue.put_nowait(audio_chunk)
                self.audio_buffer.append(audio_chunk)
            except:
                # 如果队列满，丢弃最旧的数据
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_chunk)
                    self.audio_buffer.append(audio_chunk)
                except:
                    pass
            
        except Exception as e:
            print(f"音频回调错误: {str(e)}")

    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.is_final = False
        self.audio_buffer = []
        self.text_buffer = []
        
        try:
            # 创建输入流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.stream.start()
            print("开始录音...")
            
            # 开启识别线程
            self.recognize_thread = threading.Thread(target=self._process_audio)
            self.recognize_thread.start()
            
        except Exception as e:
            print(f"启动录音失败: {str(e)}")
            self.is_recording = False

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.is_final = False
        
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            print("录音结束")
            
        except Exception as e:
            print(f"停止录音失败: {str(e)}")
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        
        # 清空缓冲区
        self.audio_buffer = []
        self.text_buffer = []
        
        if hasattr(self, 'recognize_thread'):
            self.recognize_thread.join()

    def _process_audio(self):
        """音频处理线程"""
        accumulated_chunks = []
        silence_chunks = 0
        is_speaking = False
        current_text = ""
        cache = {}  # 添加缓存字典
        
        while self.is_recording or (not self.audio_queue.empty() and is_speaking):
            try:
                # 获取音频数据
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    accumulated_chunks.append(chunk)
                    
                    # 检测是否有语音
                    energy = np.sqrt(np.mean(np.square(chunk)))
                    if energy > self.ENERGY_THRESHOLD:
                        silence_chunks = 0
                        is_speaking = True
                    else:
                        silence_chunks += 1
                    
                    # 如果启用了实时识别且积累了足够的数据进行识别
                    if self.enable_realtime_asr and self.streaming_model is not None and len(accumulated_chunks) >= self.ACCUMULATION_CHUNKS:
                        audio_data = np.concatenate(accumulated_chunks)
                        
                        # 流式识别
                        try:
                            result = self.streaming_model.generate(
                                input=audio_data,
                                cache=cache,  # 使用缓存
                                is_final=False,  # 非最终块
                                chunk_size=[0, 10, 5],  # 设置块大小
                                encoder_chunk_look_back=4,  # 编码器回看块数
                                decoder_chunk_look_back=1  # 解码器回看块数
                            )
                            
                            if result and isinstance(result, list) and len(result) > 0:
                                text = result[0].get('text', '')
                                if text:
                                    current_text = text
                                    self.partial_result.emit(text)
                        except Exception as e:
                            print(f"流式识别错误: {str(e)}")
                            # 如果出现错误，尝试重新加载模型
                            if "NoneType" in str(e):
                                print("尝试重新加载流式识别模型...")
                                try:
                                    global streaming_asr_model
                                    streaming_asr_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4", device="cuda:0", disable_progress_bar=True, disable_update=True)
                                    self.streaming_model = streaming_asr_model
                                except Exception as load_error:
                                    print(f"重新加载模型失败: {str(load_error)}")
                        
                        accumulated_chunks = []
                    
                    # 检测句子结束（静音超过一时长）
                    if is_speaking and silence_chunks >= self.SILENCE_CHUNKS:
                        is_speaking = False
                        
                        # 对整句话进行离线识别
                        try:
                            if self.audio_buffer:
                                full_audio = np.concatenate(self.audio_buffer)
                                result = self.offline_model.generate(
                                    input=full_audio,
                                    is_final=True  # 最终块
                                )
                                
                                if result and isinstance(result, list) and len(result) > 0:
                                    text = result[0].get('text', '')
                                    if text and text.strip():
                                        self.correction_result.emit(text.strip())
                                
                                # 清空缓冲区
                                self.audio_buffer = []
                                self.text_buffer = []
                                current_text = ""
                                cache = {}  # 重置缓存
                        except Exception as e:
                            print(f"离线识别错误: {str(e)}")
                
                else:
                    time.sleep(0.001)  # 短暂休眠减少 CPU 使用
                    
            except Exception as e:
                print(f"处理错误: {str(e)}")
                import traceback
                traceback.print_exc()

    def save_audio(self, filename):
        """保存录音"""
        if self.audio_buffer:
            try:
                audio_data = np.concatenate(self.audio_buffer)
                sf.write(filename, audio_data, self.sample_rate)
                print(f"录音已保存到: {filename}")
            except Exception as e:
                print(f"保存录音失败: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self, async_helper):
        super().__init__()
        self.setWindowTitle("桌面助手")
        
        # 保存异步助手实例
        self.async_helper = async_helper
        
        # 初始化组件
        self.init_ui()
        
        # 创建录音器
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.partial_result.connect(self.on_partial_result)
        self.audio_recorder.correction_result.connect(self.on_correction_result)
        self.audio_recorder.volume_update.connect(self.update_volume)
        
        # 初始化GPT客户端
        self.gpt_client = GPTClient()
        # 连接GPT信号
        self.gpt_client.sentence_received.connect(self.on_chatgpt_sentence)
        self.gpt_client.partial_response.connect(self.on_chatgpt_partial_response)
        self.gpt_client.response_received.connect(self.on_chatgpt_response)
        
        # 创建TTS客户端
        self.tts_client = TTSClient()
        # 设置异步助手
        self.tts_client.set_async_helper(self.async_helper)
        
        # 加载配置
        self.load_config()
        
        # 初始化VTS连接
        self.vts_connected = False
        self.async_helper.run_async(self.init_vts_connection())
        
        # 添加息状态跟踪
        self.current_assistant_message = None
        self.current_assistant_end = None
        self.is_assistant_message_started = False
        self.current_response_id = None

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)

        # 创建主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 创建签页
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # 创建聊天标签页
        chat_tab = QWidget()
        chat_layout = QVBoxLayout(chat_tab)
        
        # 聊天显示域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)
        
        # 音量显示
        volume_layout = QHBoxLayout()
        volume_label = QLabel("音量:")
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_bar)
        chat_layout.addLayout(volume_layout)
        
        # 开始对话按钮
        self.record_button = QPushButton("开始对话")
        self.record_button.clicked.connect(self.toggle_recording)
        chat_layout.addWidget(self.record_button)
        
        tabs.addTab(chat_tab, "聊天")
        
        # 创建测试标签页
        test_tab = QWidget()
        test_layout = QVBoxLayout(test_tab)
        
        # ASR试组
        asr_test_group = QGroupBox("ASR测试")
        asr_test_layout = QVBoxLayout(asr_test_group)
        
        # 流式识别测试钮
        test_streaming_button = QPushButton("测试流式识别")
        test_streaming_button.clicked.connect(self.test_streaming_asr)
        asr_test_layout.addWidget(test_streaming_button)
        
        # 离线试按钮
        test_offline_button = QPushButton("测试离线识别")
        test_offline_button.clicked.connect(self.test_offline_asr)
        asr_test_layout.addWidget(test_offline_button)
        
        test_layout.addWidget(asr_test_group)
        
        # TTS测试组
        tts_test_group = QGroupBox("TTS测试")
        tts_test_layout = QVBoxLayout(tts_test_group)
        
        # TTS连接测试按钮
        test_tts_button = QPushButton("测试TTS连接")
        test_tts_button.clicked.connect(self.test_tts)
        tts_test_layout.addWidget(test_tts_button)
        
        test_layout.addWidget(tts_test_group)
        
        # ChatGPT测试组
        chatgpt_test_group = QGroupBox("ChatGPT试")
        chatgpt_test_layout = QVBoxLayout(chatgpt_test_group)
        
        # ChatGPT连接测试按钮
        test_chatgpt_button = QPushButton("测试ChatGPT连接")
        test_chatgpt_button.clicked.connect(self.test_chatgpt)
        chatgpt_test_layout.addWidget(test_chatgpt_button)
        
        test_layout.addWidget(chatgpt_test_group)
        
        # 在测试标签页中添加VTS测试组
        vts_test_group = QGroupBox("VTS测试")
        vts_test_layout = QVBoxLayout(vts_test_group)
        
        # VTS按钮
        self.vts_connect_button = QPushButton("连接VTS")
        self.vts_connect_button.clicked.connect(self.toggle_vts_connection)
        vts_test_layout.addWidget(self.vts_connect_button)
        
        # VTS测试按钮
        self.vts_test_button = QPushButton("测试VTS动作")
        self.vts_test_button.clicked.connect(self.test_vts_movement)
        self.vts_test_button.setEnabled(False)  # 初始禁用
        vts_test_layout.addWidget(self.vts_test_button)
        
        # VTS状态标签
        self.vts_status_label = QLabel("未连接")
        vts_test_layout.addWidget(self.vts_status_label)
        
        test_layout.addWidget(vts_test_group)
        
        # 测试结果区域
        self.test_display = QTextEdit()
        self.test_display.setReadOnly(True)
        test_layout.addWidget(self.test_display)
        
        tabs.addTab(test_tab, "测试")
        
        # 创建配置标签页
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # API设置组
        api_group = QGroupBox("API设置")
        api_form = QFormLayout(api_group)
        
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(QLineEdit.Password)
        api_form.addRow("OpenAI API Key:", self.openai_key_input)
        
        self.openai_base_url_input = QLineEdit()
        api_form.addRow("OpenAI Base URL:", self.openai_base_url_input)
        
        self.tts_base_url_input = QLineEdit()
        api_form.addRow("TTS Base URL:", self.tts_base_url_input)
        
        # 添加模型选择
        self.model_name_input = QLineEdit()
        self.model_name_input.setText("gpt-4o")  # 设置默认值
        api_form.addRow("GPT模型:", self.model_name_input)
        
        config_layout.addWidget(api_group)
        
        # TTS置
        tts_group = QGroupBox("TTS设置")
        tts_form = QFormLayout(tts_group)
        
        self.reference_audio_input = QLineEdit()
        self.reference_audio_input.setReadOnly(True)
        reference_audio_layout = QHBoxLayout()
        reference_audio_layout.addWidget(self.reference_audio_input)
        select_audio_button = QPushButton("选择")
        select_audio_button.clicked.connect(self.select_reference_audio)
        reference_audio_layout.addWidget(select_audio_button)
        tts_form.addRow("参考音频:", reference_audio_layout)
        
        # 加TTS高级置
        self.text_split_method_combo = QComboBox()
        self.text_split_method_combo.addItems(["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"])
        self.text_split_method_combo.setCurrentText("cut5")
        tts_form.addRow("文本分割方法:", self.text_split_method_combo)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 10)
        self.batch_size_spin.setValue(1)
        tts_form.addRow("处理大小:", self.batch_size_spin)
        
        self.speed_factor_spin = QDoubleSpinBox()
        self.speed_factor_spin.setRange(0.5, 2.0)
        self.speed_factor_spin.setSingleStep(0.1)
        self.speed_factor_spin.setValue(1.0)
        tts_form.addRow("速:", self.speed_factor_spin)
        
        config_layout.addWidget(tts_group)
        
        # ASR设置组
        asr_group = QGroupBox("语音识别设置")
        asr_form = QFormLayout(asr_group)
        
        # 添加实时识别开关
        self.enable_realtime_asr = QCheckBox()
        self.enable_realtime_asr.setChecked(True)  # 默认启用
        asr_form.addRow("启动实时识别:", self.enable_realtime_asr)
        
        self.energy_threshold_input = QDoubleSpinBox()
        self.energy_threshold_input.setRange(0.001, 0.1)
        self.energy_threshold_input.setSingleStep(0.001)
        self.energy_threshold_input.setValue(0.005)
        asr_form.addRow("语音能量阈值:", self.energy_threshold_input)
        
        self.volume_threshold_input = QDoubleSpinBox()
        self.volume_threshold_input.setRange(0.001, 1.0)
        self.volume_threshold_input.setSingleStep(0.001)
        self.volume_threshold_input.setValue(0.01)
        asr_form.addRow("音量阈值:", self.volume_threshold_input)
        
        self.silence_chunks_input = QSpinBox()
        self.silence_chunks_input.setRange(5, 50)
        self.silence_chunks_input.setValue(20)
        asr_form.addRow("静音检测长度:", self.silence_chunks_input)
        
        config_layout.addWidget(asr_group)
        
        # 保存按钮
        save_button = QPushButton("保存配置")
        save_button.clicked.connect(self.save_config)
        config_layout.addWidget(save_button)
        
        tabs.addTab(config_tab, "配置")

    def toggle_recording(self):
        """切换录音状态"""
        if self.record_button.text() == "开始对话":
            self.record_button.setText("停止对话")
            self.audio_recorder.start_recording()
        else:
            self.record_button.setText("开始对话")
            self.audio_recorder.stop_recording()
    
    def on_correction_result(self, text):
        """处理语音识别纠正结果"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # 添加用户消息
        cursor.insertText("用户: " + text + "\n")
        
        # 移动到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
        # 发送消息给ChatGPT
        self.async_helper.run_async(self.gpt_client.add_message(text))
    
    def update_volume(self, volume):
        """更新音量显示"""
        # 音量值映射到0-100范围
        volume_percentage = min(100, int(volume * 1000))
        self.volume_bar.setValue(volume_percentage)
    
    async def _process_tts(self, sentence):
        """处理TTS请求"""
        try:
            success, error = await self.tts_client.text_to_speech(sentence, task_id=self.current_response_id)
            if not success:
                print(f"语音合成失败: {error}")
        except Exception as e:
            print(f"处理TTS请求时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_chatgpt_sentence(self, sentence):
        """处理完整的句子"""
        # 使用异步助手运行异步方法
        self.async_helper.run_async(self._process_tts(sentence))
    
    def on_chatgpt_response(self, response):
        """处理完整的ChatGPT响应"""
        # 在这里重置 response_id，确保所有 TTS 任务都完成
        self.response_id = None
        self.current_response = ""
        self.current_sentence = ""
        
        # 滚动到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def select_reference_audio(self):
        """选择参考音频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择参考音频", "", "WAV Files (*.wav)")
        if file_name:
            self.reference_audio_input.setText(file_name)
    
    def save_config(self):
        """保存配置"""
        config = {
            'openai_api_key': self.openai_key_input.text(),
            'openai_base_url': self.openai_base_url_input.text(),
            'tts_base_url': self.tts_base_url_input.text(),
            'reference_audio': self.reference_audio_input.text(),
            'energy_threshold': self.energy_threshold_input.value(),
            'volume_threshold': self.volume_threshold_input.value(),
            'silence_chunks': self.silence_chunks_input.value(),
            'text_split_method': self.text_split_method_combo.currentText(),
            'batch_size': self.batch_size_spin.value(),
            'speed_factor': self.speed_factor_spin.value(),
            'enable_realtime_asr': self.enable_realtime_asr.isChecked(),
            'model_name': self.model_name_input.text()
        }
        
        # 保存到.env文件
        with open('.env', 'w', encoding='utf-8') as f:
            for key, value in config.items():
                f.write(f"{key.upper()}={value}\n")
        
        # 更新录音器参数
        self.audio_recorder.ENERGY_THRESHOLD = config['energy_threshold']
        self.audio_recorder.VOLUME_THRESHOLD = config['volume_threshold']
        self.audio_recorder.SILENCE_CHUNKS = config['silence_chunks']
        self.audio_recorder.set_realtime_asr(config['enable_realtime_asr'])
        
        # 更新TTS客户端配置
        tts_base_url = config['tts_base_url'] if config['tts_base_url'] else "http://127.0.0.1:9880"
        self.tts_client.base_url = tts_base_url
        self.tts_client.ref_audio_path = config['reference_audio']
        self.tts_client.update_config(
            text_split_method=config['text_split_method'],
            batch_size=config['batch_size'],
            speed_factor=config['speed_factor']
        )
        
        # 更新GPT客户端配置
        self.gpt_client.update_config(
            api_key=config['openai_api_key'],
            base_url=config['openai_base_url'] if config['openai_base_url'] else "https://api.openai.com/v1",
            model_name=config['model_name']
        )
        
        QMessageBox.information(self, "提示", "配置已保存")
    
    def load_config(self):
        """加载配置"""
        if os.path.exists('.env'):
            load_dotenv(override=True)
            
            self.openai_key_input.setText(os.getenv('OPENAI_API_KEY', ''))
            
            # 加载 base URL,如果为空使用默认值
            openai_base_url = os.getenv('OPENAI_BASE_URL', '')
            tts_base_url = os.getenv('TTS_BASE_URL', '')
            
            self.openai_base_url_input.setText(openai_base_url)
            self.tts_base_url_input.setText(tts_base_url)
            
            self.reference_audio_input.setText(os.getenv('REFERENCE_AUDIO', ''))
            
            # 加载ASR参数
            energy_threshold = float(os.getenv('ENERGY_THRESHOLD', '0.005'))
            volume_threshold = float(os.getenv('VOLUME_THRESHOLD', '0.02'))
            silence_chunks = int(os.getenv('SILENCE_CHUNKS', '20'))
            
            self.energy_threshold_input.setValue(energy_threshold)
            self.volume_threshold_input.setValue(volume_threshold)
            self.silence_chunks_input.setValue(silence_chunks)
            
            # 更新录音器参数
            self.audio_recorder.ENERGY_THRESHOLD = energy_threshold
            self.audio_recorder.VOLUME_THRESHOLD = volume_threshold
            self.audio_recorder.SILENCE_CHUNKS = silence_chunks
            
            # 加载TTS参数
            self.text_split_method_combo.setCurrentText(os.getenv('TEXT_SPLIT_METHOD', 'cut5'))
            self.batch_size_spin.setValue(int(os.getenv('BATCH_SIZE', '1')))
            self.speed_factor_spin.setValue(float(os.getenv('SPEED_FACTOR', '1.0')))
            
            # 加载模型名称
            model_name = os.getenv('MODEL_NAME', 'gpt-4o')
            self.model_name_input.setText(model_name)
            
            # 更新TTS客户端配置
            tts_base_url = tts_base_url if tts_base_url else "http://127.0.0.1:9880"
            self.tts_client.base_url = tts_base_url
            self.tts_client.ref_audio_path = os.getenv('REFERENCE_AUDIO', '')
            self.tts_client.update_config(
                text_split_method=os.getenv('TEXT_SPLIT_METHOD', 'cut5'),
                batch_size=int(os.getenv('BATCH_SIZE', '1')),
                speed_factor=float(os.getenv('SPEED_FACTOR', '1.0'))
            )
            
            # 更新GPT客户端配置
            self.gpt_client.update_config(
                api_key=os.getenv('OPENAI_API_KEY', ''),
                base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                model_name=model_name
            )
            
            # 加载实时识别设置
            enable_realtime_asr = os.getenv('ENABLE_REALTIME_ASR', 'True').lower() == 'true'
            self.enable_realtime_asr.setChecked(enable_realtime_asr)
            self.audio_recorder.set_realtime_asr(enable_realtime_asr)  # 更新录音器的实时识别设置

    def test_streaming_asr(self):
        """测试流式识别"""
        try:
            if streaming_asr_model is None:
                self.test_display.append("流式识别模型未加载")
                return
            
            # 创建测试音频数据
            test_audio = np.zeros(800, dtype=np.float32)
            result = streaming_asr_model.generate(
                input=test_audio,
                cache={},
                is_final=False,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            self.test_display.append("流式识别测试成功")
            self.test_display.append(f"结果: {result}")
        except Exception as e:
            self.test_display.append(f"流式识别测试失败: {str(e)}")
    
    def test_offline_asr(self):
        """测试离线识别"""
        try:
            # 创建测试音频数据
            test_audio = np.zeros(800, dtype=np.float32)
            result = non_streaming_asr_model(audio_in=test_audio)
            self.test_display.append("离线识别测试成功")
            self.test_display.append(f"结果: {result}")
        except Exception as e:
            self.test_display.append(f"离线识别测试失败: {str(e)}")
    
    def test_tts(self):
        """测试TTS连接"""
        success, message = self.tts_client.test_connection()
        if success:
            self.test_display.append("TTS连接测试成功")
        else:
            self.test_display.append(f"TTS连接测试失败: {message}")
    
    async def test_chatgpt(self):
        """测试ChatGPT连接"""
        success, message = await self.gpt_client.test_connection()
        if success:
            self.test_display.append("ChatGPT连接测试成功")
            self.test_display.append(f"响应: {message}")
        else:
            self.test_display.append(f"ChatGPT连接测试失败: {message}")

    def on_partial_result(self, text):
        """处理实时识别结果"""
        # 不显示部分结果
        pass

    def on_chatgpt_partial_response(self, response, response_id=None):
        """处理ChatGPT的部分回复"""
        cursor = self.chat_display.textCursor()
        
        # 如果是新的响应ID，创建新的助手消息
        if response_id != self.current_response_id:
            self.current_response_id = response_id
            self.is_assistant_message_started = False
            self.current_assistant_message = None
            self.current_assistant_end = None
        
        # 如果是新的助手消息添加前缀
        if not self.is_assistant_message_started:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("助手: ")
            self.is_assistant_message_started = True
            self.current_assistant_message = cursor.position()
            # 保���当前文档的结束位置作为助手消息的结束位置
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.current_assistant_end = cursor.position()
        
        # 移动到助手消息的内容开始位置
        cursor.setPosition(self.current_assistant_message)
        # 选择到助手消息的结束位置
        cursor.setPosition(self.current_assistant_end, QTextCursor.MoveMode.KeepAnchor)
        
        # 替换内容
        cursor.insertText(response + "\n\n")
        
        # 更新助手消息的结束位置
        self.current_assistant_end = cursor.position()
        
        # 更新滚动条
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def toggle_vts_connection(self):
        """切换VTS连接状态"""
        try:
            if not self.vts_connected:
                # 连接 VTS
                self.async_helper.run_async(self.init_vts_connection())
            else:
                # 断开 VTS
                self.async_helper.run_async(self.tts_client.cleanup())
                self.vts_connected = False
                self.vts_connect_button.setText("连接VTS")
                self.vts_test_button.setEnabled(False)
                self.vts_status_label.setText("未连接")
        except Exception as e:
            print(f"切换VTS连接状态失败: {str(e)}")
            self.vts_status_label.setText(f"连接错误: {str(e)}")
            self.vts_connect_button.setText("连接VTS")
            self.vts_test_button.setEnabled(False)
            self.vts_connected = False

    def test_vts_movement(self):
        """测试VTS动作"""
        if not self.tts_client.vts_manager or not self.tts_client.vts_manager.is_connected:
            QMessageBox.warning(self, "警告", "VTS未连接")
            return
            
        try:
            # 向右移动
            self.async_helper.run_async(self.tts_client.vts_manager.move_model(0.5, 0, time_sec=1.0))
            QTimer.singleShot(1000, lambda: self.async_helper.run_async(self.tts_client.vts_manager.move_model(-0.5, 0, time_sec=1.0)))
            QTimer.singleShot(2000, lambda: self.async_helper.run_async(self.tts_client.vts_manager.move_model(0, 0, time_sec=1.0)))
            
            # 背颜色
            QTimer.singleShot(3000, lambda: self.async_helper.run_async(self.tts_client.vts_manager.set_background_color("#87CEEB")))
            QTimer.singleShot(4000, lambda: self.async_helper.run_async(self.tts_client.vts_manager.set_background_color("#000000")))
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"测试动作失败: {str(e)}")

    def on_vts_connection_changed(self, connected: bool):
        """VTS连接状态化回调"""
        status = "已连接" if connected else "未连接"
        self.vts_status_label.setText(status)
        self.vts_test_button.setEnabled(connected)
        print(f"VTS连接状态: {status}")

    def on_vts_error(self, error_msg: str):
        """VTS错误回调"""
        print(f"VTS错误: {error_msg}")
        self.vts_status_label.setText(f"错误: {error_msg}")
        # 尝试重新连接
        if self.tts_client.vts_manager and self.tts_client.vts_manager.is_connected:
            QTimer.singleShot(2000, self.reconnect_vts)

    def reconnect_vts(self):
        """重新连接VTS"""
        try:
            print("尝试重新连接VTS...")
            self.async_helper.run_async(self.tts_client.cleanup())
            time.sleep(1)  # 等待连接完全关闭
            self.async_helper.run_async(self.tts_client.init_vts())
        except Exception as e:
            print(f"重新连接VTS失败: {str(e)}")
            QTimer.singleShot(5000, self.reconnect_vts)

    def cleanup(self):
        """清资源"""
        if hasattr(self, 'tts_client'):
            self.async_helper.run_async(self.tts_client.cleanup())
        if hasattr(self, 'gpt_client'):
            self.gpt_client.stop()

    async def init_vts_connection(self):
        """初始化VTS连接"""
        try:
            await self.tts_client.init_vts()
            self.vts_connected = True
            self.vts_connect_button.setText("断开VTS")
            self.vts_test_button.setEnabled(True)
            self.vts_status_label.setText("已连接")
        except Exception as e:
            print(f"初始化VTS连接失败: {str(e)}")
            self.vts_connected = False
            self.vts_connect_button.setText("连接VTS")
            self.vts_test_button.setEnabled(False)
            self.vts_status_label.setText("未连接")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        
        # 创建事件循环
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
        
        # 创建异步助手
        async_helper = AsyncHelper()
        async_helper.set_loop(loop)
        
        # 创建主窗口
        window = MainWindow(async_helper)
        window.show()
        
        # 运行事件环
        with loop:
            loop.run_forever()
            
    except Exception as e:
        print(f"程序启动错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

