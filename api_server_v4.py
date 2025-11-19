# api_server.py
import uvicorn
from contextlib import asynccontextmanager
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main_v4 import app as astra_app # 导入 M2 中编译好的 LangGraph app
from dotenv import load_dotenv
from main_v4 import engine, vector_store

# 加载环境变量 (API 密钥)
load_dotenv()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # --- 这是 "启动" (startup) 时运行的代码 ---
#     print("--- API 启动 (Lifespan): 正在检查/创建 pgvector 表... ---")
#     with engine.connect() as conn:
#         # 这个 connect() 调用会触发 LlamaIndex/SQLAlchemy 
#         # 检查并创建 "astra_collection" 表
#         pass
#     print("--- pgvector 表已就绪 ---")

#     yield #  yield 之前的代码是 "startup"

#     # --- 这是 "关闭" (shutdown) 时运行的代码 ---
#     print("--- API 正在关闭... ---")

# 1. 初始化 FastAPI app
app = FastAPI(
    title="Astra Agent API (M3)",
    description="一个用于 Manus-like Agent 的流式 API",
    version="0.1.0",
    # lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # 允许 Vue 开发服务器
    allow_credentials=True,
    allow_methods=["*"], # 允许所有方法 (POST, GET...)
    allow_headers=["*"], # 允许所有 Headers
)

# 2. 定义输入模型 (Pydantic)
# 这将告诉 FastAPI 如何验证 POST 请求的 body
class TaskRequest(BaseModel):
    task: str

# 3. 定义"直播" (SSE) 生成器 (O2: 终极流式版)
async def stream_generator(task: str, session_id: str):
    """
    (O2) 这是一个使用 astream_events() 的"终极"异步生成器。
    它将"直播" LangGraph 内部的 *每一个* 事件, 包括 token。
    """
    config = {"configurable": {"thread_id": session_id}}

    # 格式: "data: <json_string>\n\n"
    def format_sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        # 关键: 我们使用 app.astream_events() (V2)
        # 这会流式返回图中发生的 *所有* 事件
        async for event in astra_app.astream_events({"task": task}, config=config, version="v2"):

            kind = event["event"]

            # 1. "on_chat_model_stream" (LLM 正在吐 token)
            # 这就是我们想要的 "逐字" 流!
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and (content := chunk.content):
                    # 我们把它包装成一个自定义的 "token" 事件
                    yield format_sse({"event": "token", "data": content})

            # 2. "on_node_start" (一个 Agent 开始工作了)
            elif kind == "on_node_start":
                node_name = event["name"]
                yield format_sse({"event": "agent_start", "node": node_name})

            # 3. (可选) "on_tool_start" (Agent 正在使用工具)
            elif kind == "on_tool_start":
                 yield format_sse({
                     "event": "tool_start", 
                     "node": event["name"], 
                     "tool": event["tags"][0] # (TavilySearch)
                 })

            # (你可以添加 on_tool_end, on_node_end 等...)

        # 4. 当流程结束时, 发送一个"结束"信号
        yield format_sse({"event": "end", "message": "Stream finished"})

    except Exception as e:
        # 5. 如果中途出错, 发送一个"错误"信号
        yield format_sse({"event": "error", "message": str(e)})

# 6. 创建我们的 API 路由 (Endpoint)
@app.post("/astra/stream")
async def run_astra_stream(request: TaskRequest):
    """
    运行 Astra Agent 并实时直播结果。
    """
    
    # (在真实世界中, session_id 应该来自用户认证)
    # (现在我们先硬编码一个, 以便 LangGraph 可以重用记忆)
    session_id = "global_api_session" 
    
    return StreamingResponse(
        stream_generator(request.task, session_id), 
        media_type="text/event-stream" # 关键! 告诉浏览器这是 SSE
    )

# 7. (可选) 添加一个根路由, 方便检查服务器是否在线
@app.get("/")
def read_root():
    return {"message": "Astra M3 API is running. Go to /docs for API."}

# 8. (用于本地运行)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)