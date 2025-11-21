import os
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import sqlalchemy
import json
from tools import python_interpreter

# --- 1. 配置与初始化 ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
web_search_tool = TavilySearch(max_results=3)

DB_NAME = "astra_db"
DB_USER = "maple" # 请确认用户名
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"

db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = sqlalchemy.create_engine(db_url)

vector_store = PGVectorStore.from_params(
    database=DB_NAME, host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
    table_name="astra_collection", embed_dim=1536
)
embed_model = OpenAIEmbedding()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=embed_model)

# --- 2. 状态定义 ---
class AgentState(TypedDict):
    task: str
    plan: List[str] # 明确类型为 List
    final_report: str
    current_step: str
    analysis_results: Annotated[List[str], operator.add] # 历史累积
    retry_count: int
    image_data: str

# --- 3. 辅助函数: 动态计算下一步 ---
def get_next_step_name(plan: List[str], current_step_name: str) -> str:
    """根据当前完成的步骤，在计划列表中找到下一步"""
    try:
        current_index = plan.index(current_step_name)
        if current_index + 1 < len(plan):
            return plan[current_index + 1]
        else:
            return "END"
    except ValueError:
        return "END"

# --- 4. 节点定义 ---

# 1. 规划师 (Planner)
async def planner_node(state: AgentState) -> dict:
    print("--- [规划师] 开始工作 ---")
    task = state.get("task")
    
    # [优化] 告诉 Planner "Write" 是可选的
    prompt = f"""
    你是一个专业的项目规划师。任务: {task}
    请制定步骤计划，从以下选择:
    1. "Research": 需要外部信息。
    2. "Analyze": 需要计算、代码执行或生成文件。
    3. "Write": 需要写一份详细的文字总结报告。
    
    【关键规则】:
    - 如果任务只是要求直接的答案、计算结果或生成特定文件(如"生成5个名字", "画张图")，**不要**包含 "Write"。
    - 只有当用户明确需要"报告"、"总结"或任务很复杂需要解释时，才包含 "Write"。
    
    请只返回 JSON 列表, 例如: ["Research", "Analyze"] 或 ["Analyze", "Write"]
    """
    messages = [HumanMessage(content=prompt)]
    
    full_plan_str = ""
    async for chunk in llm.astream(messages):
        full_plan_str += chunk.content or ""
    
    try:
        if "```json" in full_plan_str:
             full_plan_str = full_plan_str.split("```json")[1].split("```")[0]
        plan_steps = json.loads(full_plan_str.strip())
    except Exception as e:
        plan_steps = ["Research", "Write"] 
    
    # 初始步骤
    first_step = plan_steps[0] if plan_steps else "END"
    return {"plan": plan_steps, "current_step": first_step}

# 2. 研究员 (Researcher)
def researcher_node(state: AgentState) -> dict:
    print("--- [研究员] 开始搜索 ---")
    task = state.get("task")
    try:
        research_results = web_search_tool.invoke(task)
    except Exception as e:
        research_results = [f"搜索失败: {e}"]
    
    MAX_CHARS = 4000
    documents = []
    for result in research_results:
        content = str(result)
        if len(content) > MAX_CHARS: content = content[:MAX_CHARS] + "..."
        documents.append(Document(text=content, metadata={"task": task}))
    
    if documents:
        index.insert_nodes(documents)
    
    # [优化] 自动计算下一步
    plan = state.get("plan", [])
    next_step = get_next_step_name(plan, "Research")
    return {"current_step": next_step}

# 3. 分析师 (Analyst)
async def analyst_node(state: AgentState) -> dict:
    print("--- [分析师] 开始分析 ---")
    task = state.get("task")
    image_data = state.get("image_data") # 获取图片
    previous_results = state.get("analysis_results", [])
    retry_count = state.get("retry_count", 0)
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    rag_context = await query_engine.aquery(task)
    
    prompt = f"""
      你是一个精通 Python 的数据分析师。任务: {task}
      背景: {rag_context}
      要求:
      1. 务必使用 print() 输出最终结果。
      2. 仅在需要时绘图 (/app/output.png)。
      3. **不要**在代码中设置绘图风格 (如 `sns.set_theme`)，环境已预置了支持中文的学术风格配置。直接画图即可。
    """
    if retry_count > 0 and previous_results:
        prompt += f"\n【修复错误】上次失败: {previous_results[-1]}\n请修正代码。"
    
    # [M11 核心逻辑] 构造多模态消息
    if image_data:
        print("--- [分析师] 收到图片, 正在进行视觉分析... ---")
        # 如果有图片，content 是一个列表
        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            }
        ]
    else:
        # 如果没图片，content 就是普通字符串
        message_content = prompt
    
    analyst_llm = llm.bind_tools([python_interpreter])
    messages = [HumanMessage(content=message_content)]
    
    response = await analyst_llm.ainvoke(messages)
    analysis_output = response.content
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "python_interpreter":
                code = tool_call["args"]["code"]
                print(f"--- [分析师] 执行代码... ---")
                analysis_output = python_interpreter.invoke(code)

    current_retry = state.get("retry_count", 0)
    
    # [优化] 自动计算下一步
    plan = state.get("plan", [])
    next_step = get_next_step_name(plan, "Analyze")
    
    return {
        "analysis_results": [analysis_output], 
        "retry_count": current_retry + 1,
        "current_step": next_step # 更新为下一步 (可能是 Write 或 END)
    }

# 4. 写手 (Writer)
async def writer_node(state: AgentState) -> dict:
    print("--- [写手] 开始写作 ---")
    task = state.get("task")
    query_engine = index.as_query_engine(similarity_top_k=3)
    rag_context = await query_engine.aquery(task)
    analysis_results = state.get("analysis_results", [])
    
    prompt = f"""
    专业写手。任务: {task}
    上下文: {rag_context}
    数据分析: {analysis_results}
    请撰写报告。
    """
    messages = [HumanMessage(content=prompt)]
    
    full_report = ""
    async for chunk in llm.astream(messages):
        full_report += chunk.content or ""

    return {"final_report": full_report, "current_step": "END"}

# --- 5. 路由逻辑 (通用化) ---

def universal_router(state: AgentState) -> str:
    """通用路由器: 根据 current_step 直接映射到节点"""
    step = state.get("current_step")
    
    if step == "Research": return "Researcher"
    if step == "Analyze": return "Analyst"
    if step == "Write": return "Writer"
    if step == "END": return END
    return END # 默认

def qc_router(state: AgentState) -> str:
    """M10: 质量控制路由 (Analyst 专用)"""
    results = state.get("analysis_results", [])
    last_result = results[-1] if results else ""
    
    is_error = False
    try:
        data = json.loads(last_result)
        if data.get("exit_code", 0) != 0 or data.get("error"): is_error = True
    except:
        if "执行错误" in str(last_result): is_error = True

    if is_error:
        retry_count = state.get("retry_count", 0)
        if retry_count < 3:
            print(f"🔥🔥🔥 [QC] 错误, 重试第 {retry_count + 1} 次...")
            return "Analyst" # 还在 Analyst 节点闭环
        else:
            print("--- [QC] 重试耗尽 ---")
    
    # 如果成功，或者重试耗尽，则去 state 里的 current_step 指向的地方
    # (注意：analyst_node 已经把 current_step 更新为下一步了，比如 END 或 Write)
    return universal_router(state)

# --- 6. 构建图 ---
workflow = StateGraph(AgentState)

workflow.add_node("Planner", planner_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Planner")

# Planner -> Router
workflow.add_conditional_edges("Planner", universal_router)

# Researcher -> Router
workflow.add_conditional_edges("Researcher", universal_router)

# Analyst -> QC Router -> (Analyst or Next Node)
workflow.add_conditional_edges("Analyst", qc_router)

# Writer -> End
workflow.add_conditional_edges("Writer", lambda x: END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_after=["Planner"])