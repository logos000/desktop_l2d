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
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            model_revision='v2.0.4'
        )
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    while True:
        try:
            # 录制音频
            audio = record_audio()
            print("音频录制完成，开始处理...")
            
            # 保存为临时wav文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                print("保存音频文件...")
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes((audio * 32767).astype(np.int16).tobytes())
                
                # 使用FunASR进行识别
                print("开始语音识别...")
                try:
                    res = model.generate(
                        input=temp_file.name,
                        batch_size_s=300
                    )
                    print(f"\n识别结果: {res[0]['text']}\n")
                except Exception as e:
                    print(f"识别过程出错: {str(e)}")
                
                # 删除临时文件
                os.unlink(temp_file.name)
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
        
        # 询问是否继续
        choice = input("是否继续录音？(y/n): ")
        if choice.lower() != 'y':
            break

    print("程序结束")

if __name__ == "__main__":
    main() 