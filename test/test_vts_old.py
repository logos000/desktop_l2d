import logging
import time
from vts_connector.vts_manager import VTSManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def on_connection_changed(connected: bool):
    """连接状态变化回调"""
    logging.info(f"连接状态: {'已连接' if connected else '未连接'}")
    
def on_model_info_updated(model_info: dict):
    """模型信息更新回调"""
    logging.info(f"模型信息更新: {model_info}")
    
def on_error(error_msg: str):
    """错误回调"""
    logging.error(f"发生错误: {error_msg}")

def main():
    # 创建VTS管理器
    manager = VTSManager()
    
    # 设置回调函数
    manager.on_connection_changed = on_connection_changed
    manager.on_model_info_updated = on_model_info_updated
    manager.on_error = on_error
    
    try:
        # 启动管理器
        logging.info("正在启动VTS管理器...")
        manager.start()
        logging.info("VTS管理器已启动")
        
        # 等待连接建立
        logging.info("等待连接建立...")
        time.sleep(5)  # 增加等待时间
        
        if manager.is_connected:
            logging.info("成功连接到VTuber Studio")
            
            # 测试移动模型
            logging.info("测试移动模型...")
            
            # 向右移动
            logging.info("向右移动...")
            manager.move_model(0.5, 0, time_sec=1.0)
            time.sleep(2)
            
            # 向左移动
            logging.info("向左移动...")
            manager.move_model(-0.5, 0, time_sec=1.0)
            time.sleep(2)
            
            # 回到中心
            logging.info("回到中心...")
            manager.move_model(0, 0, time_sec=1.0)
            time.sleep(2)
            
            # 测试背景颜色
            logging.info("测试背景颜色...")
            manager.set_background_color("#87CEEB")  # 天蓝色
            time.sleep(2)
            manager.set_background_color("#000000")  # 黑色
            time.sleep(2)
            
            logging.info("测试完成")
            
            # 等待一段时间以便观察结果
            time.sleep(3)
        else:
            logging.error("未能连接到VTuber Studio")
            
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            # 停止管理器
            logging.info("正在停止VTS管理器...")
            manager.stop()
            logging.info("VTS管理器已停止")
        except Exception as e:
            logging.error(f"停止VTS管理器时出错: {str(e)}")

if __name__ == "__main__":
    main() 