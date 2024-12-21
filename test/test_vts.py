import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import asyncio
import wave

# 添加父目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vts_connector.vts_manager_async import VTSManagerAsync

def calculate_volume(audio_chunk):
    """计算音频块的音量"""
    return float(np.sqrt(np.mean(np.square(audio_chunk))))

def on_vts_error(error_msg: str):
    print(f"VTS错误: {error_msg}")

def on_vts_connection_changed(connected: bool):
    print(f"VTS连接状态: {'已连接' if connected else '未连接'}")

def on_vts_model_info_updated(model_info: dict):
    print(f"VTS模型信息更新: {model_info}")

async def test_vts_mouth_sync():
    """测试VTS口型同步"""
    # 初始化VTS管理器
    vts_manager = VTSManagerAsync(
        on_error=on_vts_error,
        on_connection_changed=on_vts_connection_changed,
        on_model_info_updated=on_vts_model_info_updated
    )
    
    try:
        print("正在连接VTS...")
        await vts_manager.start()
        await asyncio.sleep(1)  # 等待连接完全建立
        
        if not vts_manager.is_connected:
            print("无法连接到VTS")
            return
            
        print("VTS连接成功，开始播放测试音频...")
        
        # 读取测试音频文件
        audio_file = "skadi.wav"  # 使用 skadi.wav 进行测试
        if not os.path.exists(audio_file):
            print(f"找不到测试音频文件: {audio_file}")
            return
            
        # 使用 soundfile 打开音频文件
        with sf.SoundFile(audio_file) as audio_file:
            print(f"音频采样率: {audio_file.samplerate}")
            print(f"音频通道数: {audio_file.channels}")
            
            # 创建音频流
            with sd.OutputStream(samplerate=audio_file.samplerate, 
                               channels=audio_file.channels, 
                               dtype=np.float32) as stream:
                print("开始播放音频并同步口型...")
                
                chunk_size = 2048  # 增大块大小以减少处理频率
                volume_scale = 8.0  # 增大音量缩放以使口型变化更明显
                min_volume = 0.01   # 最小音量阈值
                
                while True:
                    # 读取一块音频数据
                    chunk = audio_file.read(chunk_size)
                    if len(chunk) == 0:
                        break
                        
                    # 转换为float32类型
                    chunk = chunk.astype(np.float32)
                    
                    # 计算音量
                    volume = calculate_volume(chunk)
                    
                    # 将音量映射到口型参数（0-1范围）
                    if volume > min_volume:
                        mouth_open = min(1.0, volume * volume_scale)
                    else:
                        mouth_open = 0.0
                    
                    # 设置口型
                    try:
                        if vts_manager.is_connected:
                            await vts_manager.set_parameter("MouthOpen", mouth_open)
                    except Exception as e:
                        print(f"设置口型参数失败: {str(e)}")
                        break
                    
                    # 播放音频
                    stream.write(chunk)
                    
                    # 等待一小段时间以同步口型和音频
                    await asyncio.sleep(0.001)
                    
        print("测试完成")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 重置口型
        try:
            if vts_manager.is_connected:
                await vts_manager.set_parameter("MouthOpen", 0)
        except:
            pass
        
        # 关闭VTS连接
        await vts_manager.stop()

if __name__ == "__main__":
    asyncio.run(test_vts_mouth_sync()) 