import asyncio
import json
import websockets
import uuid
import time
from typing import Optional, Dict, Any, Callable

class VTSManagerAsync:
    """异步版本的 VTubeStudio API 管理器"""
    
    def __init__(self, 
                 api_url: str = "ws://localhost:8001",
                 plugin_name: str = "Desktop Assistant",
                 plugin_developer: str = "Cursor AI",
                 on_error: Optional[Callable[[str], None]] = None,
                 on_connection_changed: Optional[Callable[[bool], None]] = None,
                 on_model_info_updated: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        初始化 VTS 管理器
        :param api_url: VTube Studio API WebSocket URL
        :param plugin_name: 插件名称
        :param plugin_developer: 插件开发者
        :param on_error: 错误回调函数
        :param on_connection_changed: 连接状态变化回调函数
        :param on_model_info_updated: 模型信息更新回调函数
        """
        self.api_url = api_url
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.on_error = on_error
        self.on_connection_changed = on_connection_changed
        self.on_model_info_updated = on_model_info_updated
        
        self.websocket = None
        self.is_connected = False
        self.auth_token = None
        self.model_info = {}
        self._keep_alive_task = None
        self._lock = asyncio.Lock()
        
    async def start(self):
        """启动 VTS 连接"""
        try:
            self.websocket = await websockets.connect(self.api_url)
            self.is_connected = True
            if self.on_connection_changed:
                self.on_connection_changed(True)
                
            # 认证插件
            await self._authenticate()
            
            # 获取模型信息
            await self._update_model_info()
            
            # 启动保活任务
            self._keep_alive_task = asyncio.create_task(self._keep_alive())
            
        except Exception as e:
            self.is_connected = False
            if self.on_error:
                self.on_error(str(e))
            if self.on_connection_changed:
                self.on_connection_changed(False)
            raise
            
    async def stop(self):
        """停止 VTS 连接"""
        if self._keep_alive_task:
            self._keep_alive_task.cancel()
            try:
                await self._keep_alive_task
            except asyncio.CancelledError:
                pass
            
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self.is_connected = False
        if self.on_connection_changed:
            self.on_connection_changed(False)
            
    async def _send_request(self, request_type: str, data: dict = None) -> dict:
        """发送请求到 VTS"""
        if not self.websocket:
            raise Exception("WebSocket 未连接")
            
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": request_type,
            "data": data or {}
        }
        
        async with self._lock:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            return json.loads(response)
        
    async def _authenticate(self):
        """认证插件"""
        # 获取令牌
        response = await self._send_request("AuthenticationTokenRequest", {
            "pluginName": self.plugin_name,
            "pluginDeveloper": self.plugin_developer
        })
        
        if response.get("data", {}).get("authenticationToken"):
            self.auth_token = response["data"]["authenticationToken"]
            
            # 使用令牌进行认证
            response = await self._send_request("AuthenticationRequest", {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_developer,
                "authenticationToken": self.auth_token
            })
            
            if not response.get("data", {}).get("authenticated"):
                raise Exception("认证失败")
                
    async def _update_model_info(self):
        """更新模型信息"""
        response = await self._send_request("CurrentModelRequest")
        self.model_info = response.get("data", {})
        if self.on_model_info_updated:
            self.on_model_info_updated(self.model_info)
            
    async def _keep_alive(self):
        """保持连接活跃"""
        while True:
            try:
                await asyncio.sleep(1)
                if self.websocket and self.is_connected:
                    await self._send_request("Statistics")
            except Exception as e:
                if self.on_error:
                    self.on_error(f"保活错误: {str(e)}")
                self.is_connected = False
                if self.on_connection_changed:
                    self.on_connection_changed(False)
                break
                
    async def set_parameter(self, parameter_name: str, value: float):
        """设置参数值"""
        if not self.is_connected:
            raise Exception("未连接到 VTS")
            
        await self._send_request("InjectParameterDataRequest", {
            "parameterValues": [{
                "id": parameter_name,
                "value": value
            }]
        })
        
    async def move_model(self, x: float, y: float, time_sec: float = 0.0):
        """移动模型"""
        if not self.is_connected:
            raise Exception("未连接到 VTS")
            
        await self._send_request("MoveModelRequest", {
            "timeInSeconds": time_sec,
            "valuesAreRelativeToModel": False,
            "positionX": x,
            "positionY": y
        })
        
    async def set_background_color(self, color: str):
        """设置背景颜色"""
        if not self.is_connected:
            raise Exception("未连接到 VTS")
            
        # 移除可能的 # 前缀
        color = color.lstrip("#")
        
        await self._send_request("BackgroundColorRequest", {
            "colorRGB": color
        }) 