import asyncio
import logging
import threading
from typing import Optional, Dict, Any, Callable, List
from .vts_client import VTSClient

class VTSManager:
    """VTS管理器，用于管理与VTuber Studio的连接"""
    
    def __init__(self):
        self.client = VTSClient()
        self.is_connected = False
        self.current_model_info = None
        self._event_loop = None
        self._running = False
        self._thread = None
        self._connection_lock = threading.Lock()
        
        # 设置日志
        self.logger = logging.getLogger("VTSManager")
        self.logger.setLevel(logging.INFO)
        
        # 设置回调函数
        self.client.on_connection_changed = self._on_connection_changed
        self.client.on_model_info_updated = self._on_model_info_updated
        self.client.on_error = self._on_error
        
        # 用户回调函数
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
        self.on_model_info_updated: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
    def _on_connection_changed(self, connected: bool):
        """内部连接状态变化处理"""
        with self._connection_lock:
            self.is_connected = connected
            if self.on_connection_changed:
                try:
                    self.on_connection_changed(connected)
                except Exception as e:
                    self.logger.error(f"处理连接状态变化回调时出错: {str(e)}")
            
    def _on_model_info_updated(self, model_info: Dict):
        """内部模型信息更新处理"""
        try:
            self.current_model_info = model_info
            if self.on_model_info_updated:
                self.on_model_info_updated(model_info)
        except Exception as e:
            self.logger.error(f"处理模型信息更新回调时出错: {str(e)}")
            
    def _on_error(self, error_msg: str):
        """内部错误处理"""
        try:
            self.logger.error(f"VTS错误: {error_msg}")
            if self.on_error:
                self.on_error(error_msg)
        except Exception as e:
            self.logger.error(f"处理错误回调时出错: {str(e)}")
        
    def _run_event_loop(self):
        """在新线程中运行事件循环"""
        try:
            self.logger.info("启动事件循环...")
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
            
            # 创建一个Future来控制事件循环
            self._running_future = self._event_loop.create_future()
            
            # 连接到VTS
            self.logger.info("尝试连接到VTuber Studio...")
            connect_task = self._event_loop.create_task(self.client.connect())
            self._event_loop.run_until_complete(connect_task)
            
            # 如果连接成功，运行事件循环
            if connect_task.result():
                self.logger.info("连接成功，开始运行事件循环")
                self._event_loop.run_until_complete(self._running_future)
            else:
                self.logger.error("连接失败")
                
        except Exception as e:
            self.logger.error(f"事件循环错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                if not self._event_loop.is_closed():
                    self._event_loop.close()
                    self.logger.info("事件循环已关闭")
            except Exception as e:
                self.logger.error(f"关闭事件循环时出错: {str(e)}")
            
    def start(self):
        """启动VTS管理器"""
        if not self._running:
            try:
                self.logger.info("启动VTS管理器...")
                self._running = True
                self._thread = threading.Thread(target=self._run_event_loop)
                self._thread.daemon = True
                self._thread.start()
                self.logger.info("VTS管理器启动成功")
            except Exception as e:
                self.logger.error(f"启动VTS管理器时出错: {str(e)}")
                self._running = False
                raise
                
    def stop(self):
        """停止VTS管理器"""
        if self._running:
            try:
                self.logger.info("正在停止VTS管理器...")
                self._running = False
                
                if self._event_loop and hasattr(self, '_running_future'):
                    self._event_loop.call_soon_threadsafe(
                        lambda: self._running_future.set_result(None)
                        if not self._running_future.done()
                        else None
                    )
                    
                    if self._thread:
                        self._thread.join(timeout=5)  # 等待最多5秒
                        
                    if not self._event_loop.is_closed():
                        try:
                            self._event_loop.run_until_complete(self.client.disconnect())
                        except Exception as e:
                            self.logger.error(f"断开连接时出错: {str(e)}")
                        finally:
                            self._event_loop.close()
                            
                    self._event_loop = None
                    self._thread = None
                    self.logger.info("VTS管理器已停止")
                    
            except Exception as e:
                self.logger.error(f"停止VTS管理器时出错: {str(e)}")
                raise
                
    def move_model(self, x: float, y: float, rotation: float = 0, size: float = 1, time_sec: float = 0.5):
        """移动模型"""
        if self.is_connected and self._event_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.client.move_model_position(x, y, rotation, size, time_sec),
                    self._event_loop
                )
            except Exception as e:
                self.logger.error(f"移动模型时出错: {str(e)}")
            
    def trigger_hotkey(self, hotkey_id: str):
        """触发热键"""
        if self.is_connected and self._event_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.client.trigger_hotkey(hotkey_id),
                    self._event_loop
                )
            except Exception as e:
                self.logger.error(f"触发热键时出错: {str(e)}")
            
    def set_parameter(self, parameter_name: str, value: float, weight: float = 1.0):
        """设置参数值"""
        if self.is_connected and self._event_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.client.set_parameter_value(parameter_name, value, weight),
                    self._event_loop
                )
            except Exception as e:
                self.logger.error(f"设置参数时出错: {str(e)}")
            
    def set_expression(self, expression_file: str, active: bool):
        """设置表情"""
        if self.is_connected and self._event_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.client.set_expression_state(expression_file, active),
                    self._event_loop
                )
            except Exception as e:
                self.logger.error(f"设置表情时出错: {str(e)}")
            
    def set_background_color(self, color: str):
        """设置背景颜色"""
        if self.is_connected and self._event_loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.client.set_background(color),
                    self._event_loop
                )
            except Exception as e:
                self.logger.error(f"设置背景颜色时出错: {str(e)}")
            
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        return self.current_model_info