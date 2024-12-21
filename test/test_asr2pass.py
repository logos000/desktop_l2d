import pyaudio
import numpy as np
import time
from funasr import AutoModel
import wave
import threading
from queue import Queue
import soundfile as sf

class ASR2Pass:
    def __init__(self):
        print("正在加载模型...")
        # 流式识别模型
        self.streaming_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
        # 非流式识别模型
        self.offline_model = AutoModel(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        
        # 频参数
        self.sample_rate = 16000
        self.chunk_size = 800  # 每个音频块的采样点数
        
        # 缓冲区
        self.audio_buffer = []  # 用于存储音频数据
        self.text_buffer = []   # 用于存储识别文本
        
        # 状态标志
        self.is_recording = False
        self.is_final = False
        
        # 音频队列
        self.audio_queue = Queue()
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        
        print("模型加载完成")
        
    def start_recording(self):
        """开始录��"""
        self.is_recording = True
        self.is_final = False
        self.audio_buffer = []
        self.text_buffer = []
        
        # 开启录音线程
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
        
        # 开启识别线程
        self.recognize_thread = threading.Thread(target=self._process_audio)
        self.recognize_thread.start()
        
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        self.is_final = True
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        if hasattr(self, 'recognize_thread'):
            self.recognize_thread.join()
            
    def _record_audio(self):
        """录音线程"""
        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("开始录音...")
        while self.is_recording:
            try:
                # 读取音频数据
                data = stream.read(self.chunk_size)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # 将数据放入队列
                self.audio_queue.put(audio_chunk)
                
                # 保存到缓冲区
                self.audio_buffer.append(audio_chunk)
                
            except Exception as e:
                print(f"录音错误: {str(e)}")
                break
        
        print("录音结束")
        stream.stop_stream()
        stream.close()
        
    def _process_audio(self):
        """音频处理线程"""
        accumulated_chunks = []
        silence_chunks = 0
        is_speaking = False
        
        while self.is_recording or not self.audio_queue.empty() or self.is_final:
            try:
                # 获取音频数据
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    accumulated_chunks.append(chunk)
                    
                    # 检测是否有语音
                    energy = np.sqrt(np.mean(np.square(chunk)))
                    if energy > 0.01:  # 语音阈值
                        silence_chunks = 0
                        is_speaking = True
                    else:
                        silence_chunks += 1
                    
                    # 如果积累了足够的数据，进行流式识别
                    if len(accumulated_chunks) >= 5:  # 每5个chunk进行一次识别
                        audio_data = np.concatenate(accumulated_chunks)
                        
                        # 流式识别
                        try:
                            result = self.streaming_model.generate(
                                input=audio_data,
                                speech_lengths=np.array([len(audio_data)], dtype=np.int32),
                                text=np.array([0], dtype=np.int32),
                                text_lengths=np.array([1], dtype=np.int32)
                            )
                            
                            if result and isinstance(result, list) and len(result) > 0:
                                text = result[0].get('text', '')
                                if text:
                                    print(f"[实时识别] {text}")
                                    self.text_buffer.append(text)
                        except Exception as e:
                            print(f"流式识别错误: {str(e)}")
                        
                        accumulated_chunks = []
                    
                    # 检测句子结束（静音超过一定时长）
                    if is_speaking and silence_chunks >= 10:  # 20个chunk的静音
                        is_speaking = False
                        
                        # 对整句话进行离线识别
                        try:
                            full_audio = np.concatenate(self.audio_buffer)
                            # 计算音频长度
                            speech_length = np.array([len(full_audio)], dtype=np.int32)
                            # 调用离线模型
                            result = self.offline_model.generate(
                                input=full_audio,
                                speech_lengths=speech_length,
                                text=np.array([0], dtype=np.int32),
                                text_lengths=np.array([1], dtype=np.int32)
                            )
                            
                            if result and isinstance(result, list) and len(result) > 0:
                                text = result[0].get('text', '')
                                if text:
                                    print(f"[离线识别] {text}")
                                    
                            # 清空缓冲区
                            self.audio_buffer = []
                            self.text_buffer = []
                            
                        except Exception as e:
                            print(f"离线识别错误: {str(e)}")
                
                elif self.is_final:
                    # 最后的离线识别
                    if self.audio_buffer:
                        try:
                            full_audio = np.concatenate(self.audio_buffer)
                            # 计算音频长度
                            speech_length = np.array([len(full_audio)], dtype=np.int32)
                            # 调用离线模型
                            result = self.offline_model.generate(
                                input=full_audio,
                                speech_lengths=speech_length,
                                text=np.array([0], dtype=np.int32),
                                text_lengths=np.array([1], dtype=np.int32)
                            )
                            
                            if result and isinstance(result, list) and len(result) > 0:
                                text = result[0].get('text', '')
                                if text:
                                    print(f"[最终识别] {text}")
                        except Exception as e:
                            print(f"最终识别错误: {str(e)}")
                    break
                    
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

def main():
    asr = ASR2Pass()
    
    try:
        asr.start_recording()
        print("按Enter键停止录音...")
        input()
        asr.stop_recording()
        
        # 保存录音
        asr.save_audio("recording.wav")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        asr.stop_recording()

if __name__ == "__main__":
    main() 