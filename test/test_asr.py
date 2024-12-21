import pyaudio
import numpy as np
import time
from funasr import AutoModel
import wave

def record_audio(filename, duration=5, sample_rate=16000):
    """录制音频并保存为WAV文件"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=1024)
    
    print(f"开始录音，持续{duration}秒...")
    frames = []
    
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    print("录音结束")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 保存为WAV文件
    audio_data = np.concatenate(frames, axis=0)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)  # float32 = 4 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return audio_data

def print_audio_info(audio_data, chunk_size=None):
    """打印音频数据信息"""
    print(f"\n音频数据信息:")
    print(f"- 数据类型: {audio_data.dtype}")
    print(f"- 数据形状: {audio_data.shape}")
    print(f"- 数据范围: [{audio_data.min()}, {audio_data.max()}]")
    print(f"- 均值: {audio_data.mean()}")
    print(f"- 标准差: {audio_data.std()}")
    if chunk_size:
        print(f"- 总chunk数: {len(audio_data) // chunk_size}")

def main():
    # 初始化模型
    print("正在加载模型...")
    streaming_model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
    print("预热模型...")
    dummy_input = np.zeros(800, dtype=np.float32)
    print("\n预热输入信息:")
    print_audio_info(dummy_input)
    streaming_model.generate(input=dummy_input)
    
    # 录制音频
    wav_file = "test_recording.wav"
    audio_data = record_audio(wav_file)
    
    # 打印音频信息
    chunk_size = 800 * 2
    print_audio_info(audio_data, chunk_size)
    
    # 按chunk处理音频
    is_final = False
    online_text = ""
    
    print("\n开始识别...")
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            is_final = True
        
        try:
            # 打印chunk信息
            print(f"\nChunk {i//chunk_size + 1}:")
            print_audio_info(chunk)
            
            # 计算音频长度
            speech_length = np.array([len(chunk)], dtype=np.int32)
            
            # 调用模型
            result = streaming_model.generate(
                input=chunk,
                speech_lengths=speech_length,
                text=np.array([0], dtype=np.int32),  # 占位符
                text_lengths=np.array([1], dtype=np.int32)  # 占位符
            )
            
            # 处理结果
            print(f"识别结果: {result}")
            if result:
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get('text', '')
                    if text:
                        online_text += text
                        print(f"实时识别: {text}")
        
        except Exception as e:
            print(f"识别错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n最终识别结果: {online_text}")

if __name__ == "__main__":
    main() 