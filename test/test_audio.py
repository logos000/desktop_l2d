import sounddevice as sd
import numpy as np
import soundfile as sf
import wave
import time
import pyaudio

def test_sounddevice_play():
    """测试使用sounddevice播放音频"""
    print("\n=== 测试 sounddevice 播放 ===")
    
    # 生成测试音频 (1秒的1000Hz正弦波)
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 1000 * t)
    
    # 确保音频数据是float32类型
    audio_data = audio_data.astype(np.float32)
    
    print("播放设备信息:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} ({device['hostapi']})")
        print(f"   最大输出通道: {device['max_output_channels']}")
        print(f"   默认采样率: {device['default_samplerate']}")
    
    # 查找AirPods设备
    airpods_device = None
    for i, device in enumerate(devices):
        if "AirPods" in device['name'] and device['max_output_channels'] > 0:
            airpods_device = i
            break
    
    if airpods_device is not None:
        print(f"\n使用设备 {airpods_device}: {devices[airpods_device]['name']}")
        try:
            print("\n播放测试音频...")
            # 明确指定设备
            sd.default.device = airpods_device
            sd.play(audio_data, sample_rate, blocking=True)
            print("播放完成")
        except Exception as e:
            print(f"播放出错: {str(e)}")
    else:
        print("未找到AirPods设备")

def test_pyaudio_play():
    """测试使用PyAudio播放音频"""
    print("\n=== 测试 PyAudio 播放 ===")
    
    # 生成测试音频
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 1000 * t)
    
    # 转换为float32
    audio_data = audio_data.astype(np.float32)
    
    try:
        p = pyaudio.PyAudio()
        
        # 显示音频设备信息
        print("播放设备信息:")
        airpods_device = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"{i}: {device_info['name']}")
            print(f"   最大输出通道: {device_info['maxOutputChannels']}")
            print(f"   默认采样率: {device_info['defaultSampleRate']}")
            
            # 查找AirPods设备
            if "AirPods" in device_info['name'] and device_info['maxOutputChannels'] > 0:
                airpods_device = i
        
        if airpods_device is not None:
            print(f"\n使用设备 {airpods_device}: {p.get_device_info_by_index(airpods_device)['name']}")
            # 打开音频流，指定设备
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=airpods_device
            )
            
            print("\n播放测试音频...")
            # 播放音频
            stream.write(audio_data.tobytes())
            
            # 清理
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("播放完成")
        else:
            print("未找到AirPods设备")
            p.terminate()
        
    except Exception as e:
        print(f"播放出错: {str(e)}")

def test_wav_file():
    """测试播放WAV文件"""
    print("\n=== 测试播放 WAV 文件 ===")
    
    # 首先创建一个测试WAV文件
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 1000 * t)
    
    # 保存为WAV文件
    test_wav = "test_audio.wav"
    with wave.open(test_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    try:
        # 使用soundfile读取
        print("\n使用soundfile读取并播放...")
        audio_data, sample_rate = sf.read(test_wav)
        
        # 查找AirPods设备
        devices = sd.query_devices()
        airpods_device = None
        for i, device in enumerate(devices):
            if "AirPods" in device['name'] and device['max_output_channels'] > 0:
                airpods_device = i
                break
        
        if airpods_device is not None:
            print(f"\n使用设备 {airpods_device}: {devices[airpods_device]['name']}")
            sd.default.device = airpods_device
            sd.play(audio_data, sample_rate, blocking=True)
            print("播放完成")
        else:
            print("未找到AirPods设备")
        
    except Exception as e:
        print(f"播放出错: {str(e)}")

if __name__ == "__main__":
    # 测试不同的播放方法
    test_sounddevice_play()
    time.sleep(1)  # 等待1秒
    
    test_pyaudio_play()
    time.sleep(1)  # 等待1秒
    
    test_wav_file() 