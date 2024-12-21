import numpy as np
import sounddevice as sd
import wave
import tempfile

def record_audio(duration=5, sample_rate=16000):
    """录制音频"""
    print(f"录音开始，持续 {duration} 秒...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    print("录音结束")
    return audio_data

def save_wav(audio_data, filename, sample_rate=16000):
    """保存为WAV文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

def main():
    # 录制音频
    audio = record_audio()
    
    # 保存为WAV文件
    output_file = "test.wav"
    save_wav(audio, output_file)
    print(f"音频已保存到: {output_file}")

if __name__ == "__main__":
    main() 