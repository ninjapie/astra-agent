# api_server.py
import uvicorn
from contextlib import asynccontextmanager
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main_v6 import app as astra_app # 导入 M2 中编译好的 LangGraph app
from dotenv import load_dotenv
from main_v6 import engine, vector_store

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

# 3. 定义"直播" (SSE) 生成器 (O3: 终极流式版)
async def stream_generator(task: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    
    def format_sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        # 仍然使用 astream_events
        async for event in astra_app.astream_events({"task": task}, config=config, version="v2"):
            
            kind = event["event"]
            node_name = event.get("name") # 获取当前节点名

            # 1. "on_chat_model_stream" (LLM 正在吐 token)
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and (content := chunk.content):
                    
                    # (!! O3 关键 !!) 我们用 "name" 来区分流
                    if node_name == "Planner":
                        # 规划师正在"流式"思考
                        yield format_sse({"event": "plan_token", "data": content})
                    
                    elif node_name == "Writer":
                        # 写手正在"流式"写作
                        yield format_sse({"event": "report_token", "data": content})

            # 2. "on_node_start" (一个 Agent 开始工作了)
            elif kind == "on_node_start":
                # 过滤掉内部节点 (比如主管路由)
                if not node_name or node_name.startswith("__"):
                    continue
                yield format_sse({"event": "agent_start", "node": node_name})

            # 3. "on_tool_start" 
            elif kind == "on_tool_start":
                 yield format_sse({
                     "event": "tool_start", 
                     "node": event["name"], 
                     "tool": event["tags"][0] 
                 })
            
            # (!! O3 关键 !!) 监听 "on_node_end" 来获取 Planner 的"最终"计划
            elif kind == "on_node_end":
                if node_name == "Planner":
                    # Planner 流式结束后, 我们把"最终"解析好的 plan 发送出去
                    final_plan = event["data"]["output"].get("plan")
                    if final_plan and isinstance(final_plan, list):
                        yield format_sse({"event": "plan_final", "data": final_plan})

            # 4. 结束/错误
            elif kind == "end":
                yield format_sse({"event": "end", "message": "Stream finished"})
            elif kind == "error":
                yield format_sse({"event": "error", "message": str(event["data"])})
            
    except Exception as e:
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