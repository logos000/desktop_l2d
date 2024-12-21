import os
import numpy as np
import sounddevice as sd
import wave
import tempfile
from funasr import AutoModel

def record_audio(duration=5, sample_rate=16000):
    """录制音频"""
    print(f"录音开始，持续 {duration} 秒...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("录音结束")
    return audio_data

def main():
    # 初始化语音识别模型
    print("正在加载模型...")
    try:
        model = AutoModel(
            model="paraformer-zh-streaming",
            model_revision='v2.0.4'
        )
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 设置流式识别参数
    chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms
    encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
    decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

    while True:
        try:
            # 录制音频
            audio = record_audio()
            print("音频录制完成，开始处理...")
            
            # 使用流式识别
            print("开始流式识别...")
            chunk_stride = chunk_size[1] * 960  # 600ms
            total_chunk_num = int(len(audio - 1) / chunk_stride + 1)
            cache = {}
            recognized_text = ""
            
            print(f"音频将被分成 {total_chunk_num} 个片段进行处理")
            for i in range(total_chunk_num):
                print(f"处理第 {i+1}/{total_chunk_num} 个片段...")
                speech_chunk = audio[i*chunk_stride:(i+1)*chunk_stride]
                is_final = i == total_chunk_num - 1
                res = model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back
                )
                if res and len(res) > 0:
                    recognized_text += res[0]["text"]
                    print(f"当前识别结果: {res[0]['text']}")
            
            print(f"\n最终识别结果: {recognized_text}\n")
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
        
        # 询问是否继续
        choice = input("是否继续录音？(y/n): ")
        if choice.lower() != 'y':
            break

    print("程序结束")

if __name__ == "__main__":
    main() 