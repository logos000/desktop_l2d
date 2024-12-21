import os
import sys
from pathlib import Path

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
try:
    from gpt_service.tools.screenshot import screenshot_tool
except ImportError:
    from tools.screenshot import screenshot_tool
import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
import json
try:
    from gpt_service.gpt_client import GPTClient
except ImportError:
    from gpt_client import GPTClient

class StreamingCallbackHandler(BaseCallbackHandler):
    """流式输出回调处理器"""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """处理新的token"""
        print(token, end="", flush=True)

def num_tokens_from_messages(messages, model="gpt-4o"):
    """计算消息的token数量"""
    encoding = tiktoken.encoding_for_model("gpt-4")  # 使用 gpt-4 的编码器
    num_tokens = 0
    for message in messages:
        # 计算每条消息的基础token（根据角色）
        num_tokens += 4  # 每条消息的开头token
        
        # 计算内容的token
        if isinstance(message.content, str):
            num_tokens += len(encoding.encode(message.content))
        elif isinstance(message.content, list):
            for content in message.content:
                if content["type"] == "text":
                    num_tokens += len(encoding.encode(content["text"]))
                elif content["type"] == "image_url":
                    num_tokens += 85  # 每张图片大约85个tokens
                    
        # 计算角色的token
        num_tokens += len(encoding.encode(message.type))
    
    return num_tokens

def get_tool_description(tool):
    """获取工具的描述息"""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": tool.args
        }
    }

def test_token_usage():
    """测试token使用情况"""
    # 加载环境变量
    load_dotenv()
    
    try:
        # 获取截图
        print("\n1. 获取截图...")
        image_url = screenshot_tool.invoke({})
        print(f"[DEBUG] 第一次截图URL: {image_url}")
        print(f"截图获取并上传成功: {image_url}")
        
        # 创建流式输出回调处理器
        streaming_handler = StreamingCallbackHandler()
        
        # 创建GPT客户端
        gpt_client = GPTClient()
        
        # 1. 测试直接发送图片
        print("\n2. 测试直接发送图片...")
        direct_messages = [
            SystemMessage(content="你是一个友好的中文AI助手。请详细描述你看到的屏幕内容。"),
            HumanMessage(content=[
                {"type": "text", "text": "请分析这张截图"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                }
            ])
        ]
        
        print(f"[DEBUG] 发送给GPT的消息内容:")
        print(json.dumps(direct_messages[1].content, ensure_ascii=False, indent=2))
        
        direct_tokens = num_tokens_from_messages(direct_messages)
        print(f"直接发送图片的token数量: {direct_tokens}")
        
        # 直接调用 GPT 分析图片
        print("\n开始分析图片...")
        chat = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=gpt_client.api_key,
            openai_api_base=gpt_client.base_url,
            streaming=True,
            callbacks=[streaming_handler]
        )
        response = chat.invoke(direct_messages)
        print(f"GPT 分析结果: {response.content}")
        
        # 2. 测试使用Agent
        print("\n3. 测试使用Agent...")
        
        # 计算系统提示的token
        template_tokens = num_tokens_from_messages([SystemMessage(content=gpt_client.system_message)])
        print(f"系统提示的token数量: {template_tokens}")
        
        # 计算工具描述的token
        tool_description = json.dumps(get_tool_description(screenshot_tool), ensure_ascii=False)
        print(f"[DEBUG] 工具描述:")
        print(tool_description)
        
        encoding = tiktoken.encoding_for_model("gpt-4o")  # 使用 gpt-4 的编码器
        tool_tokens = len(encoding.encode(tool_description))
        print(f"工具描述的token数量: {tool_tokens}")
        
        # 构建完整的Agent消息
        agent_message = {
            "input": "请帮我分析当前屏幕内容",
            "chat_history": []
        }
        
        # 计算完整请求的预估token
        total_tokens = template_tokens + tool_tokens
        print(f"\n预估总token数量: {total_tokens}")
        print(f"其中:")
        print(f"- 系统提示token: {template_tokens}")
        print(f"- 工具描述token: {tool_tokens}")
        
        # 尝试执行（可能会失败）
        try:
            print("\n4. 尝试执行Agent...")
            # 添加调试信息
            print("\n调试消息内容:")
            from langchain.callbacks import get_openai_callback
            with get_openai_callback() as cb:
                try:
                    # 打印实际发送到 OpenAI 的消息
                    print("\n准备发送的消息:")
                    print(json.dumps(agent_message, ensure_ascii=False, indent=2))
                    
                    # 创建测试执行器
                    agent_executor = gpt_client.create_test_executor(streaming_handler)
                    
                    # 执行测试前记录当前截图URL
                    print(f"[DEBUG] 执行Agent前的图片URL: {image_url}")
                    
                    # 执行测试
                    response = agent_executor.invoke(agent_message)
                    
                    # 获取Agent执行后的新截图URL
                    if "take_screenshot" in str(response):
                        print("[DEBUG] Agent执行过程中调用了take_screenshot")
                        # 从输出中提取新的URL
                        import re
                        urls = re.findall(r'https://i\.ibb\.co/\w+/\w+\.png', str(response))
                        if urls:
                            print(f"[DEBUG] Agent获取的新截图URL: {urls[0]}")
                    
                    print(f"\nAgent 分析结果: {response['output']}")
                    print(f"\nToken 使用情况: {cb}")
                except Exception as e:
                    print(f"执行期间的 Token 使用: {cb}")
                    raise e
            print("执行成功！")
        except Exception as e:
            print(f"执行失败: {str(e)}")
            print("\n完整错误信息:")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_token_usage() 