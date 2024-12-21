import asyncio
import logging
from vts_client import VTSClient

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_vts_connection():
    """测试VTS连接"""
    # 使用端口8001
    client = VTSClient()
    
    # 测试连接
    connected = await client.connect()
    if not connected:
        logging.error("连接VTuber Studio失败")
        return
        
    try:
        # 获取当前模型信息
        model_info = await client.get_current_model_info()
        if model_info:
            logging.info(f"当前模型信息: {model_info}")
            
        # 获取可用的热键列表
        hotkeys = await client.get_available_hotkeys()
        if hotkeys:
            logging.info(f"可用的热键列表: {hotkeys}")
            
        # 获取可用的输入参数
        parameters = await client.get_input_parameters()
        if parameters:
            logging.info(f"可用的输入参数: {parameters}")
            
        # 获取可用的表情列表
        expressions = await client.get_expressions()
        if expressions:
            logging.info(f"可用的表情列表: {expressions}")
            
        # 测试移动模型
        logging.info("测试移动模型...")
        # 向右移动
        await client.move_model_position(0.5, 0, time_sec=1.0)
        await asyncio.sleep(1.5)
        # 向左移动
        await client.move_model_position(-0.5, 0, time_sec=1.0)
        await asyncio.sleep(1.5)
        # 回到中心
        await client.move_model_position(0, 0, time_sec=1.0)
        logging.info("模型移动测试完成")
        
        # 测试背景颜色
        logging.info("测试更改背景颜色...")
        await client.set_background("#87CEEB")  # 天蓝色
        await asyncio.sleep(1)
        await client.set_background("#000000")  # 恢复黑色
        logging.info("背景颜色测试完成")
            
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
    finally:
        # 断开连接
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_vts_connection()) 