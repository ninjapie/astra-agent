# api_server.py
import uvicorn
from contextlib import asynccontextmanager
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main_v3 import app as astra_app # 导入 M2 中编译好的 LangGraph app
from dotenv import load_dotenv
from main_v3 import engine, vector_store

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

# 3. 定义"直播" (SSE) 生成器
async def stream_generator(task: str, session_id: str):
    """
    这是一个"异步生成器", 它将"直播" LangGraph 的每一步。
    它使用 Server-Sent Events (SSE) 格式。
    """
    
    # 为此会话设置配置 (我们用它来跟踪 LangGraph 的"记忆")
    config = {"configurable": {"thread_id": session_id}}
    
    # 关键: 我们使用 app.astream() (异步流)
    try:
        # astream() 会"流式"返回图中每一步的输出
        async for step in astra_app.astream({"task": task}, config=config):
            # step 格式是: {"Node_Name": {"output_key": "output_value"}}
            
            # 将步骤数据格式化为 SSE (Server-Sent Event)
            # 格式: "data: <json_string>\n\n"
            data_to_send = json.dumps(step)
            yield f"data: {data_to_send}\n\n"
            
            # 模拟一点延迟, 让我们在前端能看清
            await asyncio.sleep(0.1) 

        # 4. 当流程结束时, 发送一个"结束"信号
        end_signal = {"event": "end", "message": "Stream finished"}
        yield f"data: {json.dumps(end_signal)}\n\n"
        
    except Exception as e:
        # 5. 如果中途出错, 发送一个"错误"信号
        error_signal = {"event": "error", "message": str(e)}
        yield f"data: {json.dumps(error_signal)}\n\n"

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