import os
import uuid
import json
import logging
from openai import OpenAI
from PySide6.QtCore import QObject, Signal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image, ImageGrab
from io import BytesIO
import base64
try:
    from gpt_service.tools.screenshot import screenshot_tool
except ImportError:
    from tools.screenshot import screenshot_tool

def get_tool_description(tool):
    """获取工具的描述信息"""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": tool.args
        }
    }

class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调处理器"""
    def __init__(self, gpt_client):
        self.gpt_client = gpt_client
        self.is_tool_call = False
        
    def on_llm_start(self, *args, **kwargs):
        """开始生成时的回调"""
        self.is_tool_call = False
        
    def on_tool_start(self, *args, **kwargs):
        """工具调用开始时的回调"""
        self.is_tool_call = True
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """处理新的token"""
        if not token or self.is_tool_call:
            return
            
        self.gpt_client.current_response += token
        
        # 发送部分响应和响应ID
        self.gpt_client.partial_response.emit(
            self.gpt_client.current_response, 
            self.gpt_client.response_id
        )
        
        # 将当前内容添加到当前句子
        self.gpt_client.current_sentence += token
        
        # 检查是否形成完整句子（包含标点符号）
        if any(p in token for p in ['。', '！', '？', '.', '!', '?', '\n']):
            if self.gpt_client.current_sentence.strip():
                self.gpt_client.sentence_received.emit(
                    self.gpt_client.current_sentence.strip(),
                    self.gpt_client.response_id
                )
                self.gpt_client.current_sentence = ""

class GPTClient(QObject):
    """GPT服务客户端"""
    response_received = Signal(str)  # 响应信号
    sentence_received = Signal(str, str)  # 完整句子信号，包含句子和响应ID
    partial_response = Signal(str, str)  # 部分响应信号，包含响应ID
    
    def __init__(self):
        super().__init__()
        # 设置日志记录器
        self.logger = logging.getLogger("GPTClient")
        self.logger.setLevel(logging.INFO)
        
        # 从环境变量载配置
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.model_name = os.getenv('MODEL_NAME', 'gpt-4o')
        
        # 初始化状态
        self.current_response = ""
        self.current_sentence = ""
        self.response_id = None
        
        # 创建回调处理器
        self.callback_handler = StreamingCallbackHandler(self)
        
        # 初始化 ChatOpenAI
        self.chat = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            streaming=True,
            callbacks=[self.callback_handler]
        )
        
        # 设置系统提示
        self.system_message = """你是一个友好的中文AI助手。请用简短、自然的中文对话方式回应用户。

当用户询问关于屏幕内容、查看屏幕、或者需要了解当前界面的情况时，你应该使用 take_screenshot 工具来获取并分析屏幕截图。

在分析截图时，请详细描述你看到的内容，包括：
1. 程序类型（如浏览器、编辑器、终端等）
2. 窗口布局和主要区域的位置
3. 具体的文本内容、代码片段或要信息
4. 任何特殊的视觉元素或状态指示器

请避免使用模糊或通用的描述，而是提供准确和有价值的信息。如果截图不清晰或无法获取，请告知用户。"""
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建代理
        self.agent = create_openai_tools_agent(
            llm=self.chat,
            tools=[screenshot_tool],
            prompt=self.prompt
        )
        
        # 设置截图工具的GPT客户端引用
        screenshot_tool.gpt_client = self
        
        # 创建执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[screenshot_tool],
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
        # 初始化对话历史
        self.chat_history = []
    
    def create_test_executor(self, streaming_handler=None):
        """创建用于测试的执行器"""
        # 创建测试用的ChatOpenAI实例
        chat = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            streaming=True,
            callbacks=[streaming_handler] if streaming_handler else None
        )
        
        print("[DEBUG] 创建Agent时使用的系统提示:")
        print(self.system_message)
        
        # 创建代理
        agent = create_openai_tools_agent(
            llm=chat,
            tools=[screenshot_tool],
            prompt=self.prompt
        )
        
        print("[DEBUG] 创建Agent时使用的工具:")
        print(json.dumps(get_tool_description(screenshot_tool), ensure_ascii=False, indent=2))
        
        # 创建执行器
        executor = AgentExecutor(
            agent=agent,
            tools=[screenshot_tool],
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[streaming_handler] if streaming_handler else None
        )
        
        return executor
    
    def add_message_with_image(self, text: str, image_url: str):
        """添加带图片的新消息并获取响应"""
        try:
            # 生成新的响应ID
            self.response_id = str(uuid.uuid4())
            
            # 重置当前响应
            self.current_response = ""
            self.current_sentence = ""
            
            # 创建OpenAI客户端，增加超时时间
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0  # 增加超时时间到30秒
            )
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": self.system_message
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # 添加重试机制
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # 使用gpt-4o模型发送请求
                    stream = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1000,
                        stream=True
                    )
                    
                    response_text = ""
                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            response_text += token
                            self.current_response = response_text
                            
                            # 发送部分响应
                            self.partial_response.emit(
                                self.current_response,
                                self.response_id
                            )
                            
                            # 处理句子
                            self.current_sentence += token
                            if any(p in token for p in ['。', '！', '？', '.', '!', '?', '\n']):
                                if self.current_sentence.strip():
                                    self.sentence_received.emit(self.current_sentence.strip(), self.response_id)
                                    self.current_sentence = ""
                    
                    # 如果成功处理，跳出重试循环
                    break
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    self.logger.warning(f"第{retry_count}次重试失败: {str(e)}")
                    if retry_count < max_retries:
                        import time
                        time.sleep(2)  # 等待2秒后重试
                    continue
            
            # 如果所有重试都失败了
            if retry_count == max_retries:
                raise Exception(f"重试{max_retries}次后仍然失败: {str(last_error)}")
            
            # 更新对话历史
            self.chat_history.extend([
                HumanMessage(content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]),
                AIMessage(content=response_text)
            ])
            
            # 发送完整响应
            self.response_received.emit(response_text)
            
        except Exception as e:
            error_msg = f"处理带图片的消息时出错: {str(e)}"
            self.logger.error(error_msg)
            self.response_received.emit(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
    
    async def handle_screenshot(self, image_base64: str, text: str):
        """处理截图数据"""
        try:
            # 生成新的响应ID
            self.response_id = str(uuid.uuid4())
            
            # 重置当前响应
            self.current_response = ""
            self.current_sentence = ""
            
            # 构建带图片的消息
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ]
            
            # 创建OpenAI客户端
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 直接调用ChatCompletion
            stream = client.chat.completions.create(
                model=self.model_name,  # 使用配置的模型名称
                messages=messages,
                stream=True,
                max_tokens=2000
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.current_response += content
                    self.current_sentence += content
                    
                    # 检查是否有完整的句子
                    if any(self.current_sentence.endswith(p) for p in ["。", "！", "？", ".", "!", "?"]):
                        self.sentence_received.emit(self.current_sentence, self.response_id)
                        self.current_sentence = ""
                    
                    # 发送部分响应
                    self.partial_response.emit(self.current_response, self.response_id)
            
            # 发送完整响应
            self.response_received.emit(self.current_response)
            
        except Exception as e:
            self.logger.error(f"处理截图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.response_received.emit(f"错误: {str(e)}")
    
    async def add_message(self, text: str, image_url: str = None):
        """添加新消息并获取响应"""
        if image_url:
            return await self.add_message_with_image(text, image_url)
                
        try:
            # 生成新的响应ID
            self.response_id = str(uuid.uuid4())
            
            # 重置当前响应
            self.current_response = ""
            self.current_sentence = ""
            
            # 执行代理
            self.logger.info("开始处理消息...")
            response = await self.agent_executor.ainvoke(
                {
                    "input": text,
                    "chat_history": self.chat_history
                }
            )
            
            # 更新对话历史
            self.chat_history.extend([
                HumanMessage(content=text),
                AIMessage(content=response["output"])
            ])
            
            # 发送完整响应
            self.response_received.emit(response["output"])
                
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.response_received.emit(f"错误: {str(e)}")
    
    async def _handle_screenshot_request(self, text: str):
        """处理截图请求"""
        try:
            self.logger.info("开始获取屏幕截图...")
            
            # 获取屏幕截图
            screenshot = ImageGrab.grab(all_screens=True)
            self.logger.info(f"成功获取截图，原始尺寸: {screenshot.size}")
            
            # 调整图片大小
            max_size = (1024, 768)
            original_width, original_height = screenshot.size
            ratio = min(max_size[0] / original_width, max_size[1] / original_height)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            screenshot = screenshot.resize(new_size, Image.LANCZOS)
            self.logger.info(f"调整后的尺寸: {new_size}")
            
            # 转换为base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG", optimize=True, quality=95)
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            self.logger.info("图片已转换为base64格式")
            
            # 保存本地副本
            try:
                os.makedirs("temp_images", exist_ok=True)
                local_path = os.path.join("temp_images", "latest_screenshot.png")
                screenshot.save(local_path, "PNG")
                self.logger.info(f"截图已保存到本地: {local_path}")
            except Exception as e:
                self.logger.warning(f"保存本地副本失败: {str(e)}")
            
            # 构建请求
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ]
            
            # 创建OpenAI客户端
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # 发送请求并获取流式响应
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=2000
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.current_response += content
                    self.current_sentence += content
                    
                    # 检查是否有完整的句子
                    if any(self.current_sentence.endswith(p) for p in ["。", "！", "？", ".", "!", "?"]):
                        self.sentence_received.emit(self.current_sentence, self.response_id)
                        self.current_sentence = ""
                    
                    # 发送部分响应
                    self.partial_response.emit(self.current_response, self.response_id)
            
            # 发送完整响应
            self.response_received.emit(self.current_response)
            
            # 更新对话历史
            self.chat_history.extend([
                HumanMessage(content=text),
                AIMessage(content=self.current_response)
            ])
            
        except Exception as e:
            error_msg = f"处理截图请求时出错: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            self.response_received.emit(error_msg)
    
    def update_config(self, api_key=None, base_url=None, model_name=None):
        """更新配置"""
        if api_key:
            self.api_key = api_key
        if base_url:
            self.base_url = base_url
        if model_name:
            self.model_name = model_name
        
        # 重新初始化 ChatOpenAI
        self.chat = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            streaming=True,
            callbacks=[self.callback_handler]
        )
        
        # 重新创建代理
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_tools_agent(
            llm=self.chat,
            tools=[screenshot_tool],
            prompt=self.prompt
        )
        
        # 重新创建执行器
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[screenshot_tool],
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
        # 重置对话历史
        self.chat_history = []
        
        self.logger.info("GPT配置已更新")
    
    async def test_connection(self):
        """测试连接"""
        try:
            response = await self.chat.ainvoke([HumanMessage(content="测试消息")])
            return True, response.content
        except Exception as e:
            self.logger.error(f"连接测试失败: {str(e)}")
            return False, str(e)
            
    def stop(self):
        """停止服务并清理资源"""
        try:
            # 重置状态
            self.current_response = ""
            self.current_sentence = ""
            self.response_id = None
            
            # 清空对话历史
            self.chat_history = []
            
            # 记录日志
            self.logger.info("GPT服务已停止")
            
        except Exception as e:
            self.logger.error(f"停止服务时出错: {str(e)}")
            import traceback
            traceback.print_exc()