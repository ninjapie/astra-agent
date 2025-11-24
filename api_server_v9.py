# api_server.py
import uvicorn
from contextlib import asynccontextmanager
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from main_v13 import app as astra_app # 导入 M2 中编译好的 LangGraph app
from dotenv import load_dotenv
from main_v13 import engine, vector_store
from fastapi.staticfiles import StaticFiles
import os
from typing import Optional

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

os.makedirs("static", exist_ok=True)
# 这样，放在 static/ 里的文件就可以通过 http://localhost:8000/static/文件名 访问了
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. 定义输入模型 (Pydantic)
# 这将告诉 FastAPI 如何验证 POST 请求的 body
class TaskRequest(BaseModel):
    task: Optional[str] = None # 如果是继续，task 可以为空
    thread_id: str # (关键) 前端生成的会话 ID
    action: str = "start" # "start" (新任务) 或 "continue" (批准/继续)
    # [M11 新增] 图片 Base64
    image: Optional[str] = None

async def stream_generator(request: TaskRequest):
    # 使用前端传来的 thread_id
    config = {"configurable": {"thread_id": request.thread_id}}
    
    def format_sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        if request.action == "start":
            if not request.task:
                yield format_sse({"event": "error", "message": "Start action requires a task"})
                return
            # [M11] 将图片放入 inputs
            inputs = {
                "task": request.task,
                "image_data": request.image # 传递图片
            }
        elif request.action == "continue":
            inputs = None
        else:
             yield format_sse({"event": "error", "message": "Invalid action"})
             return

        async for event in astra_app.astream_events(inputs, config=config, version="v2"):
            
            kind = event["event"]
            
            # --- [O5 关键修复] ---
            # 不要只看 event["name"]，它可能是 "ChatOpenAI"
            # 我们要看 metadata 里的 "langgraph_node" 来确定是哪个节点在运行
            metadata = event.get("metadata", {})
            current_node = metadata.get("langgraph_node")
            
            # 备用：如果是 on_node_start/end，name 字段通常就是节点名
            event_name = event.get("name")
            # ---------------------

            # 1. LLM 逐字流 (思考过程)
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and (content := chunk.content):
                    # 使用 current_node 来判断是谁在思考
                    if current_node == "Planner":
                        yield format_sse({"event": "plan_token", "data": content})
                    elif current_node == "Writer":
                        yield format_sse({"event": "report_token", "data": content})
                    elif current_node == "Analyst":
                         yield format_sse({"event": "analyst_token", "data": content})

            # 2. 节点开始
            elif kind == "on_node_start":
                if event_name and not event_name.startswith("__"): 
                    yield format_sse({"event": "agent_start", "node": event_name})

            # 3. 工具开始
            elif kind == "on_tool_start":
                 yield format_sse({
                     "event": "tool_start", 
                     "node": current_node or "Analyst", # 如果是在 Analyst 节点里调用的
                     "tool": event_name if event_name else "unknown"
                 })

            # 4. 工具结束 (捕获输出)
            elif kind == "on_tool_end":
                if event_name == "python_interpreter":
                    output = str(event["data"].get("output"))
                    yield format_sse({"event": "tool_output", "node": "Analyst", "output": output})
            
            # 5. 节点结束 (捕获最终计划)
            elif kind == "on_node_end":
                if event_name == "Planner":
                    # 这里的 event_name 确实是 "Planner"
                    final_plan = event["data"]["output"].get("plan")
                    if final_plan:
                        yield format_sse({"event": "plan_final", "data": final_plan})
            
            # 6. 错误处理
            elif kind == "error":
                 yield format_sse({"event": "error", "message": str(event["data"])})
            

        # 检查中断状态
        snapshot = astra_app.get_state(config)
        if snapshot.next:
            print(f"--- 流程暂停, 等待批准. 下一步: {snapshot.next} ---")
            yield format_sse({
                "event": "interrupt", 
                "next_step": snapshot.next
            })
        else:
            yield format_sse({"event": "end", "message": "Stream finished"})
            
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
    # session_id = "global_api_session" 
    import uuid
    session_id = f"session-{uuid.uuid4()}"
    
    return StreamingResponse(
        stream_generator(request), 
        media_type="text/event-stream" # 关键! 告诉浏览器这是 SSE
    )

# --- M7 修复: 专用下载路由 (处理文件名复原) ---
@app.get("/download/{filename}")
async def download_file(filename: str):
    # 1. 找到真实文件路径
    file_path = os.path.join("static", filename)

    # 2. 检查文件是否存在
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    # 3. 还原原始文件名 (去掉 UUID 前缀)
    # 假设格式是: uuid-uuid-uuid_original.xlsx
    if "_" in filename:
        # 只分割第一个 "_", 保留后面可能存在的 "_"
        original_name = filename.split("_", 1)[1]
    else:
        original_name = filename

    # 4. 返回文件，并强制浏览器以 original_name 下载
    return FileResponse(
        path=file_path, 
        filename=original_name # <--- 这就是魔法所在!
    )

# 7. (可选) 添加一个根路由, 方便检查服务器是否在线
@app.get("/")
def read_root():
    return {"message": "Astra M3 API is running. Go to /docs for API."}

# 8. (用于本地运行)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)