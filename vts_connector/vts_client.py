import websockets
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable

class VTSClient:
    """VTuber Studio API客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.plugin_name = "Desktop_Live2D_Controller"
        self.plugin_developer = "MyDesktop"
        self.authentication_token: Optional[str] = None
        
        # 设置日志
        self.logger = logging.getLogger("VTSClient")
        self.logger.setLevel(logging.INFO)
        
        # 回调函数
        self.on_connection_changed: Optional[Callable[[bool], None]] = None
        self.on_model_info_updated: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
    @property
    def ws_url(self) -> str:
        """获取WebSocket URL"""
        return f"ws://{self.host}:{self.port}"
        
    async def connect(self) -> bool:
        """连接到VTuber Studio"""
        try:
            self.logger.info(f"正在连接到 {self.ws_url}...")
            try:
                self.websocket = await asyncio.wait_for(
                    websockets.connect(self.ws_url),
                    timeout=5.0  # 5秒超时
                )
            except asyncio.TimeoutError:
                self.logger.error("连接超时")
                if self.on_error:
                    self.on_error("连接超时")
                if self.on_connection_changed:
                    self.on_connection_changed(False)
                return False
            except ConnectionRefusedError:
                self.logger.error("连接被拒绝，请确保VTuber Studio已启动并开启了API")
                if self.on_error:
                    self.on_error("连接被拒绝，请确保VTuber Studio已启动并开启了API")
                if self.on_connection_changed:
                    self.on_connection_changed(False)
                return False
                
            self.logger.info(f"已连接到VTuber Studio: {self.ws_url}")
            
            # 进行API认证
            self.logger.info("开始API认证...")
            try:
                auth_result = await asyncio.wait_for(
                    self.authenticate(),
                    timeout=5.0  # 5秒超时
                )
            except asyncio.TimeoutError:
                self.logger.error("认证超时")
                if self.on_error:
                    self.on_error("认证超时")
                await self.disconnect()
                if self.on_connection_changed:
                    self.on_connection_changed(False)
                return False
                
            if not auth_result:
                self.logger.error("API认证失败")
                await self.disconnect()
                if self.on_connection_changed:
                    self.on_connection_changed(False)
                return False
                
            self.logger.info("API认证成功")
            if self.on_connection_changed:
                self.on_connection_changed(True)
            return True
            
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            if self.on_error:
                self.on_error(f"连接失败: {str(e)}")
            if self.on_connection_changed:
                self.on_connection_changed(False)
            return False
            
    async def disconnect(self):
        """断开与VTuber Studio的连接"""
        if self.websocket:
            try:
                self.logger.info("正在断开连接...")
                # 取消所有正在进行的任务
                for task in asyncio.all_tasks(asyncio.get_event_loop()):
                    if task != asyncio.current_task():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                # 关闭websocket连接
                await self.websocket.close()
                self.websocket = None
                self.logger.info("已断开连接")
                if self.on_connection_changed:
                    self.on_connection_changed(False)
            except Exception as e:
                self.logger.error(f"断开连接时出错: {str(e)}")
                if self.on_error:
                    self.on_error(f"断开连接时出错: {str(e)}")
                if self.on_connection_changed:
                    self.on_connection_changed(False)
            
    async def authenticate(self) -> bool:
        """进行API认证"""
        if not self.websocket:
            return False
            
        try:
            # 第一步：请求认证令牌
            auth_data = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "authentication",
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.plugin_developer,
                    "pluginIcon": ""
                }
            }
            
            await self.websocket.send(json.dumps(auth_data))
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("messageType") == "AuthenticationTokenResponse":
                self.authentication_token = response_data.get("data", {}).get("authenticationToken")
                if not self.authentication_token:
                    self.logger.error("获取认证令牌失败")
                    return False
                    
                # 第二���：使用令牌进行认证
                auth_request = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "authentication",
                    "messageType": "AuthenticationRequest",
                    "data": {
                        "pluginName": self.plugin_name,
                        "pluginDeveloper": self.plugin_developer,
                        "authenticationToken": self.authentication_token
                    }
                }
                
                await self.websocket.send(json.dumps(auth_request))
                auth_response = await self.websocket.recv()
                auth_response_data = json.loads(auth_response)
                
                if auth_response_data.get("messageType") == "AuthenticationResponse":
                    if auth_response_data.get("data", {}).get("authenticated"):
                        self.logger.info("API认证成功")
                        return True
                        
            self.logger.error("认证失败")
            return False
            
        except Exception as e:
            self.logger.error(f"认证过程出错: {str(e)}")
            if self.on_error:
                self.on_error(f"认证过程出错: {str(e)}")
            return False
            
    async def send_request(self, message_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送API请求"""
        if not self.websocket or not self.authentication_token:
            self.logger.error("未连接或未认证")
            return None
            
        try:
            request = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": message_type,
                "messageType": message_type,
                "data": {
                    **data,
                    "authenticationToken": self.authentication_token
                }
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"发送请求失败: {str(e)}")
            if self.on_error:
                self.on_error(f"发送请求失败: {str(e)}")
            return None
            
    async def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        response = await self.send_request("CurrentModelRequest", {})
        if response and self.on_model_info_updated:
            self.on_model_info_updated(response)
        return response
        
    async def move_model_position(self, x: float, y: float, rotation: float = 0, size: float = 1, time_sec: float = 0.5) -> Optional[Dict[str, Any]]:
        """移动模型位置
        
        Args:
            x: X坐标 (-1.0 到 1.0)
            y: Y坐标 (-1.0 到 1.0)
            rotation: 旋转角度（弧度）
            size: 缩放比例 (0.1 到 2.0)
            time_sec: 动画时间（秒）
        """
        data = {
            "timeInSeconds": time_sec,
            "valuesAreRelativeToModel": False,
            "positionX": x,
            "positionY": y,
            "rotation": rotation,
            "size": size
        }
        return await self.send_request("MoveModelRequest", data)
        
    async def trigger_hotkey(self, hotkey_id: str) -> Optional[Dict[str, Any]]:
        """触发热键"""
        data = {
            "hotkeyID": hotkey_id
        }
        return await self.send_request("HotkeyTriggerRequest", data)
        
    async def get_available_hotkeys(self) -> Optional[List[Dict[str, Any]]]:
        """获取可用的热键列表"""
        response = await self.send_request("HotkeysInCurrentModelRequest", {})
        if response and "data" in response:
            return response["data"].get("availableHotkeys", [])
        return None
        
    async def get_input_parameters(self) -> Optional[List[Dict[str, Any]]]:
        """获取可用的输入参数列表"""
        response = await self.send_request("InputParameterListRequest", {})
        if response and "data" in response:
            return response["data"].get("parameters", [])
        return None
        
    async def set_parameter_value(self, parameter_name: str, value: float, weight: float = 1.0) -> Optional[Dict[str, Any]]:
        """设置参数值
        
        Args:
            parameter_name: 参数名称
            value: 参数值 (0.0 到 1.0)
            weight: 权重 (0.0 到 1.0)
        """
        data = {
            "id": parameter_name,
            "value": value,
            "weight": weight
        }
        return await self.send_request("InjectParameterDataRequest", data)
        
    async def set_expression_state(self, expression_file: str, active: bool) -> Optional[Dict[str, Any]]:
        """设置表情状态
        
        Args:
            expression_file: 表情文件名
            active: 是否激活
        """
        data = {
            "expressionFile": expression_file,
            "active": active
        }
        return await self.send_request("ExpressionStateRequest", data)
        
    async def get_expressions(self) -> Optional[List[Dict[str, Any]]]:
        """获取可用的表情列表"""
        response = await self.send_request("ExpressionStateRequest", {})
        if response and "data" in response:
            return response["data"].get("expressions", [])
        return None
        
    async def set_background(self, color: str) -> Optional[Dict[str, Any]]:
        """设置背景颜色
        
        Args:
            color: 颜色值，格式为"#RRGGBB"
        """
        data = {
            "backgroundColor": color,
        }
        return await self.send_request("BackgroundColorRequest", data) 