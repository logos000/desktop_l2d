import asyncio
import io
import wave
import numpy as np
import sounddevice as sd
import requests
import threading
from queue import Queue
import traceback
import logging
from typing import Optional, Dict, Any, Tuple
from vts_connector.vts_manager_async import VTSManagerAsync

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TTSClient:
    def __init__(self, base_url="http://127.0.0.1:9880"):
        self.base_url = base_url.rstrip('/')
        self.ref_audio_path = ""
        self.text_split_method = "cut5"
        self.batch_size = 1
        self.speed_factor = 1.0
        self.vts_manager = None  # VTS管理器实例
        self.async_helper = None  # 异步助手实例
        self.is_playing = False  # 是否正在播放
        self.play_lock = asyncio.Lock()  # 播放锁
        self.tts_queue = Queue()  # TTS请求队列（使用线程安全的Queue）
        self.audio_queue = asyncio.Queue()  # 音频播放队列
        self.is_processing = False  # 是否正在处理TTS队列
        self.is_playing_queue = False  # 是否正在处理播放队列
        self.tts_thread = None  # TTS处理线程
        self.current_task_id = None  # 当前任务ID
        self.current_stream = None  # 当前音频流
        self.current_chunks = None  # 当前音频块
        self.current_chunk_index = 0  # 当前音频块索引
        self.mouth_params = None  # 口型参数
        
        # 设置日志记录器
        self.logger = logging.getLogger("TTSClient")
        self.logger.setLevel(logging.INFO)
        
    def start_tts_processing(self):
        """启动TTS处理线程"""
        if self.tts_thread is None or not self.tts_thread.is_alive():
            self.is_processing = True
            self.tts_thread = threading.Thread(target=self._process_tts_queue)
            self.tts_thread.daemon = True
            self.tts_thread.start()
            self.logger.info("TTS处理线程已启动")
            
    def stop_tts_processing(self):
        """停止TTS处理线程"""
        self.is_processing = False
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join()
            self.logger.info("TTS处理线程已停止")
            
    def _process_tts_queue(self):
        """处理TTS请求队列（在线程中运行）"""
        while self.is_processing:
            try:
                # 获取下一个请求（阻塞等待）
                task = self.tts_queue.get()
                if task is None:  # 空任务表示停止处理
                    break
                    
                text, task_id = task
                self.logger.info(f"处理TTS请求: {text[:30]}... (任务ID: {task_id})")
                
                # 如果任务ID不匹配，跳过此任务
                if task_id != self.current_task_id:
                    self.logger.info(f"跳过过期任务: {task_id}")
                    continue
                
                try:
                    # 发送TTS请求
                    params = {
                        "text": text,
                        "text_lang": "zh",
                        "prompt_lang": "zh",
                        "ref_audio_path": self.ref_audio_path,
                        "text_split_method": self.text_split_method,
                        "batch_size": self.batch_size,
                        "speed_factor": self.speed_factor,
                        "media_type": "wav"
                    }
                    
                    self.logger.info("发送TTS请求...")
                    self.logger.debug(f"请求参数: {params}")
                    
                    # 发送请求
                    response = requests.get(f"{self.base_url}/tts", params=params, stream=True)
                    
                    self.logger.info(f"响应状态码: {response.status_code}")
                    self.logger.debug(f"响应头: {response.headers}")
                    
                    if response.status_code == 200:
                        # 处理音频数据
                        buffer = io.BytesIO()
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                buffer.write(chunk)
                        
                        buffer.seek(0)
                        
                        try:
                            # 解析WAV文件头
                            with wave.open(buffer, 'rb') as wav_file:
                                self.logger.info("WAV头解析成功:")
                                self.logger.debug(f"- 采样率: {wav_file.getframerate()} Hz")
                                self.logger.debug(f"- 通道数: {wav_file.getnchannels()}")
                                self.logger.debug(f"- 采样宽度: {wav_file.getsampwidth()} bytes")
                                self.logger.debug(f"- 总帧数: {wav_file.getnframes()}")
                                
                                # 获取音频数据
                                audio_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                                # 转为float32格式
                                audio_data = audio_data.astype(np.float32) / 32768.0
                                
                                # 创建音频信息字典
                                audio_info = {
                                    'audio_data': audio_data,
                                    'sample_rate': wav_file.getframerate(),
                                    'channels': wav_file.getnchannels(),
                                    'task_id': task_id  # 添加任务ID
                                }
                                
                                # 使用异步助手将音频数据加入播放队列
                                if self.async_helper:
                                    self.async_helper.run_async(self._add_to_audio_queue(audio_info))
                                    self.logger.info("音频数据已放入播放队列")
                                
                        except Exception as e:
                            self.logger.error(f"WAV解析错误: {str(e)}")
                            traceback.print_exc()
                    else:
                        error_msg = f"错误: HTTP状态码 {response.status_code}"
                        self.logger.error(error_msg)
                        self.logger.error(f"响应内容: {response.text}")
                        
                except Exception as e:
                    self.logger.error(f"处理TTS请求时出错: {str(e)}")
                    traceback.print_exc()
                finally:
                    # 标记当前请求处理完成
                    self.tts_queue.task_done()
                    
            except Exception as e:
                self.logger.error(f"TTS队列处理错误: {str(e)}")
                traceback.print_exc()
                
    async def _add_to_audio_queue(self, audio_info):
        """将音频数据添加到播放队列"""
        await self.audio_queue.put(audio_info)
        # 如果播放队列处理器未运行，启动它
        if not self.is_playing_queue:
            self.logger.info("启动音频队列处理器")
            asyncio.create_task(self.process_audio_queue())
            
    async def text_to_speech(self, text, task_id=None):
        """将文本加入TTS请求队列"""
        if not self.ref_audio_path:
            self.logger.error("未设置参考音频文件")
            return False, "未设置参考音频文件"
            
        # 如果收到新的任务ID，且与当前任务ID不同
        if task_id is not None and task_id != self.current_task_id:
            self.logger.info(f"收到新的任务ID: {task_id}，停止当前任务: {self.current_task_id}")
            await self.stop_current_task()
            self.current_task_id = task_id
            
        # 将请求加入队列
        self.tts_queue.put((text, task_id))
        self.logger.info(f"文本已加入TTS队列: {text[:30]}...")
        
        # 如果处理线程未运行，启动它
        if not self.is_processing:
            self.start_tts_processing()
            
        return True, None
            
    async def cleanup(self):
        """清理资源"""
        self.logger.info("开始清理资源...")
        # 停止TTS处理线程
        self.stop_tts_processing()
        # 停止VTS连接
        if self.vts_manager and self.vts_manager.is_connected:
            await self.vts_manager.stop()
            self.vts_manager = None
        self.logger.info("资源清理完成")

    async def init_vts(self):
        """初始化VTS连接"""
        if self.vts_manager is None:
            self.logger.info("初始化VTS连接...")
            self.vts_manager = VTSManagerAsync()
            await self.vts_manager.start()
            await asyncio.sleep(0.1)  # 等待连接建立
            self.logger.info(f"VTS连接状态: {self.vts_manager.is_connected}")

    def set_async_helper(self, helper):
        """设置异步助手"""
        self.async_helper = helper
        self.logger.info("异步助手已设置")
        
    def update_config(self, text_split_method=None, batch_size=None, speed_factor=None):
        """更新配置"""
        if text_split_method is not None:
            self.text_split_method = text_split_method
        if batch_size is not None:
            self.batch_size = batch_size
        if speed_factor is not None:
            self.speed_factor = speed_factor
        self.logger.info("配置已更新")
            
    def test_connection(self):
        """测试TTS服务器连接"""
        try:
            self.logger.info("测试TTS服务器连接...")
            # 使用 /tts 端点进行测试，发送一个简单的请求
            params = {
                "text": "测试",
                "text_lang": "zh",
                "prompt_lang": "zh",
                "ref_audio_path": self.ref_audio_path
            }
            response = requests.get(f"{self.base_url}/tts", params=params)
            if response.status_code == 200:
                self.logger.info("TTS服务器连接成功")
                return True, "连接成功"
            else:
                error_msg = f"服务器返回错误状态码: {response.status_code}"
                if response.headers.get('content-type') == 'application/json':
                    error_msg += f", 错误信息: {response.json().get('message', '')}"
                self.logger.error(error_msg)
                return False, error_msg
        except Exception as e:
            error_msg = f"连接失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
            
    def audio_callback(self, outdata, frames, time, status):
        """音频播放回调函数"""
        if status:
            self.logger.error(f'音频回调状态: {status}')
        
        if not self.is_playing or self.current_chunks is None:
            self.logger.debug("播放已停止或没有音频数据，填充静音")
            outdata.fill(0)
            return
            
        try:
            if self.current_chunk_index < len(self.current_chunks):
                # 获取当前音频块
                chunk = self.current_chunks[self.current_chunk_index]
                # 如果音频块小于需要的帧数，补充静音
                if len(chunk) < frames:
                    self.logger.debug(f"音频块{self.current_chunk_index}长度不足，补充静音")
                    temp = np.zeros(frames, dtype=np.float32)
                    temp[:len(chunk)] = chunk
                    chunk = temp
                # 确保数据形状正确
                if chunk.ndim == 1:
                    chunk = chunk[:frames].reshape(-1, 1)
                else:
                    chunk = chunk[:frames]
                outdata[:] = chunk
                
                # 设置口型
                if self.vts_manager and self.vts_manager.is_connected and self.mouth_params:
                    try:
                        mouth_open = self.mouth_params[self.current_chunk_index]
                        if self.async_helper:
                            self.async_helper.run_async(
                                self.vts_manager.set_parameter("MouthOpen", mouth_open)
                            )
                    except Exception as e:
                        self.logger.error(f"设置口型参数失败: {str(e)}")
                
                self.current_chunk_index += 1
                
                # 如果是最后一块，准备结束播放
                if self.current_chunk_index >= len(self.current_chunks):
                    self.logger.info("音频播放完成")
                    # 重置口型
                    if self.vts_manager and self.vts_manager.is_connected and self.async_helper:
                        self.async_helper.run_async(
                            self.vts_manager.set_parameter("MouthOpen", 0)
                        )
            else:
                self.logger.debug("没有更多音频数据���填充静音")
                outdata.fill(0)
                
        except Exception as e:
            self.logger.error(f"音频回调处理错误: {str(e)}")
            outdata.fill(0)
            traceback.print_exc()

    async def process_audio_queue(self):
        """处理音频播放队列"""
        self.logger.info("启动音频队列处理器")
        self.is_playing_queue = True
        
        try:
            while self.is_playing_queue:
                try:
                    # 获取下一个音频数据（这里会阻塞直到有新数据）
                    self.logger.info("等待新的音频数据...")
                    audio_info = await self.audio_queue.get()
                    self.logger.info(f"获取到新的音频数据，任务ID: {audio_info['task_id']}")
                    
                    # 检查任务ID是否匹配
                    if audio_info['task_id'] != self.current_task_id:
                        self.logger.info(f"跳过过期音频: 当前任务ID={self.current_task_id}, 音频任务ID={audio_info['task_id']}")
                        continue
                    
                    try:
                        audio_data = audio_info['audio_data']
                        sample_rate = audio_info['sample_rate']
                        channels = audio_info['channels']
                        
                        self.logger.info(f"准备播放音频: 采样率={sample_rate}Hz, 通道数={channels}, 数据长度={len(audio_data)}")
                        
                        # 设置播放状态
                        self.is_playing = True
                        
                        # 预处理音频块
                        chunk_size = int(sample_rate * 0.1)  # 每块0.1秒
                        self.current_chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
                        self.current_chunk_index = 0
                        
                        # 预计算口型参数
                        volume_scale = 5.0
                        min_volume = 0.01
                        smoothing_factor = 0.5
                        last_mouth_open = 0.0
                        self.mouth_params = []
                        
                        self.logger.debug(f"开始计算口型参数，音频块数量: {len(self.current_chunks)}")
                        for chunk in self.current_chunks:
                            volume = float(np.sqrt(np.mean(np.square(chunk))))
                            if volume > min_volume:
                                target_mouth_open = min(1.0, volume * volume_scale)
                            else:
                                target_mouth_open = 0.0
                            
                            mouth_open = (smoothing_factor * last_mouth_open + 
                                        (1 - smoothing_factor) * target_mouth_open)
                            last_mouth_open = mouth_open
                            self.mouth_params.append(mouth_open)
                        
                        # 创建并启动音频流
                        try:
                            self.current_stream = sd.OutputStream(
                                samplerate=sample_rate,
                                channels=channels,  # 使用原始通道数
                                dtype=np.float32,
                                blocksize=chunk_size,  # 使用与分块相同的大小
                                callback=self.audio_callback,
                                finished_callback=self.stream_finished_callback
                            )
                            self.current_stream.start()
                            self.logger.info("音频流已启动")
                            
                            # 等待播放完成
                            while self.is_playing and self.current_chunk_index < len(self.current_chunks):
                                await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            self.logger.error(f"创建或使用音频流时出错: {str(e)}")
                            raise
                            
                    except Exception as e:
                        self.logger.error(f"播放音频时出错: {str(e)}")
                        traceback.print_exc()
                    finally:
                        # 清理资源
                        if self.current_stream is not None:
                            self.current_stream.stop()
                            self.current_stream.close()
                            self.current_stream = None
                        
                        # 重置状态
                        self.is_playing = False
                        self.current_chunks = None
                        self.current_chunk_index = 0
                        self.mouth_params = None
                        
                        self.logger.info("播放状态已重置")
                        # 标记当前音频处理完成
                        self.audio_queue.task_done()
                        self.logger.info("音频队列任务已完成")
                    
                except asyncio.CancelledError:
                    self.logger.info("音频播放被取消")
                    break
                except Exception as e:
                    self.logger.error(f"音频播放队列处理错误: {str(e)}")
                    traceback.print_exc()
                    continue
                    
        finally:
            self.logger.info("音频队列处理器结束")
            self.is_playing_queue = False

    def stream_finished_callback(self):
        """音频流播放完成回调"""
        self.logger.info("音频流播放完成")
        self.is_playing = False

    async def stop_current_task(self):
        """停止当前TTS任务"""
        self.logger.info("停止当前TTS任务...")
        
        # 记录当前状态
        self.logger.info(f"当前状态: is_playing={self.is_playing}, is_playing_queue={self.is_playing_queue}")
        
        # 停止当前的音频流
        if self.current_stream is not None:
            try:
                self.current_stream.stop()
                self.current_stream.close()
                self.current_stream = None
                self.logger.info("成功停止音频流")
            except Exception as e:
                self.logger.error(f"停止音频流时出错: {str(e)}")
        
        # 重置播放状态和数据
        self.is_playing = False
        self.current_chunks = None
        self.current_chunk_index = 0
        self.mouth_params = None
        
        # 清空TTS请求队列
        queue_size = self.tts_queue.qsize()
        self.logger.info(f"清空TTS请求队列，当前队列大小: {queue_size}")
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except Exception as e:
                self.logger.error(f"清空TTS队列时出错: {str(e)}")
                break
        
        # 清空音频播放队列
        try:
            audio_queue_size = self.audio_queue.qsize()
            self.logger.info(f"清空音频播放队列，当前队列大小: {audio_queue_size}")
            while not self.audio_queue.empty():
                try:
                    await self.audio_queue.get()
                    self.audio_queue.task_done()
                except Exception as e:
                    self.logger.error(f"清空音频队列时出错: {str(e)}")
                    break
        except Exception as e:
            self.logger.error(f"获取音频队列大小时出错: {str(e)}")
        
        # 重置口型
        if self.vts_manager and self.vts_manager.is_connected:
            try:
                await self.vts_manager.set_parameter("MouthOpen", 0)
                self.logger.info("成功重置口型参数")
            except Exception as e:
                self.logger.error(f"重置口型失败: {str(e)}")
        
        self.logger.info("TTS任务已停止，所有状态已重置")