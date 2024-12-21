from typing import Dict, Optional, Any
import base64
from PIL import ImageGrab, Image
from io import BytesIO
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
import logging
import traceback
import asyncio
import aiohttp
from openai import OpenAI
from pydantic import Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScreenshotTool")

async def upload_to_imgbb(image_base64: str) -> str:
    """上传图片到imgbb并返回URL"""
    # 加载环境变量
    load_dotenv()
    
    # 获取API密钥
    api_key = os.getenv('IMGBB_API_KEY')
    if not api_key:
        raise Exception("请设置IMGBB_API_KEY环境变量")
    
    # 准备请求
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": api_key,
        "image": image_base64,
    }
    
    # 发送请求
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                response.raise_for_status()  # 检查响应状态
                data = await response.json()
                logger.info(f"图片上传成功，URL: {data['data']['url']}")
                return data["data"]["url"]
    except Exception as e:
        logger.error(f"上传图片失败: {str(e)}")
        raise Exception(f"上传图片失败: {str(e)}")

class ScreenshotTool(BaseTool):
    name: str = "take_screenshot"
    description: str = "获取当前屏幕的截图。当用户询问屏幕内容或需要查看屏幕时使用此工具"
    gpt_client: Optional[Any] = Field(default=None, exclude=True)
    
    def _capture_screenshot(self) -> str:
        """捕获屏幕截图并返回base64编码"""
        try:
            # 获取屏幕截图
            screenshot = ImageGrab.grab(all_screens=True)
            logger.info(f"成功获取截图，原始尺寸: {screenshot.size}")
            
            # 调整图片大小
            max_size = (1024, 768)
            original_width, original_height = screenshot.size
            ratio = min(max_size[0] / original_width, max_size[1] / original_height)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            screenshot = screenshot.resize(new_size, Image.LANCZOS)
            logger.info(f"调整后的尺寸: {new_size}")
            
            # 转换为base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG", optimize=True, quality=95)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info("图片已转换为base64格式")
            
            # 保存本地副本
            try:
                os.makedirs("temp_images", exist_ok=True)
                local_path = os.path.join("temp_images", "latest_screenshot.png")
                screenshot.save(local_path, "PNG")
                logger.info(f"截图已保存到本地: {local_path}")
            except Exception as e:
                logger.warning(f"保存本地副本失败: {str(e)}")
            
            return image_base64
            
        except Exception as e:
            logger.error(f"截图失败: {str(e)}")
            raise Exception(f"截图失败: {str(e)}")
    
    async def _analyze_screenshot(self, image_base64: str, query: str) -> str:
        """使用OpenAI分析截图"""
        try:
            # 创建OpenAI客户端
            client = OpenAI(
                api_key=self.gpt_client.api_key,
                base_url=self.gpt_client.base_url
            )
            
            # 构建专门的截图分析系统提示
            system_prompt = """你是一个专业的图像分析助手。请仔细分析用户提供的屏幕截图，并根据用户的具体问题提供相关的描述和分析。

在描述时，请注意：
1. 重点关注与用户问题相关的内容
2. 使用清晰、准确的语言
3. 如果发现用户关注的内容，要详细描述
4. 如果没有找到用户关注的内容，要明确说明

请始终使用中文回答，不要使用任何其他语言。即使用户用英文提问，也要用中文回答。
回答要简洁、准确、易懂。"""
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"请分析这张屏幕截图，重点回答：{query}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ]
            
            # 发送请求
            response = client.chat.completions.create(
                model=self.gpt_client.model_name,
                messages=messages,
                stream=False,  # 不使用流式输出
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"分析截图失败: {str(e)}")
            raise Exception(f"分析截图失败: {str(e)}")
    
    def _run(self, query: str = "") -> str:
        """同步执行截图操作"""
        raise NotImplementedError("请使用异步版本")
    
    async def _arun(self, query: str = "") -> str:
        """异步执行截图操作"""
        try:
            # 捕获屏幕截图
            image_base64 = self._capture_screenshot()
            
            # 分析截图
            result = await self._analyze_screenshot(image_base64, query)
            
            return result
            
        except Exception as e:
            logger.error(f"执行截图工具失败: {str(e)}")
            return f"抱歉，截图操作失败: {str(e)}"

# 导出工具
screenshot_tool = ScreenshotTool()