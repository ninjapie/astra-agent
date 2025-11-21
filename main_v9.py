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
    # [新增] 是否需要人工审批
    needs_review: bool

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

    同时，请判断该任务是否需要【用户审批】(needs_review):
    - 如果任务有风险（如删除文件）、非常复杂、或者指令模糊不确定，请设为 true。
    - 如果任务很简单、明确（如"画个正弦函数图"、"搜索今天的汇率"），请设为 false (自动执行)。
    
    【重要】请严格返回以下 JSON 格式（不要 Markdown，纯 JSON）:
    {{
        "plan": ["Research", "Analyze"],
        "needs_review": false
    }}
    """
    messages = [HumanMessage(content=prompt)]
    
    full_plan_str = ""
    async for chunk in llm.astream(messages):
        full_plan_str += chunk.content or ""
    
    try:
        # 清理 Markdown 标记
        clean_str = full_plan_str.replace("```json", "").replace("```", "").strip()
        result_data = json.loads(clean_str)
        
        plan_steps = result_data.get("plan", ["Research", "Write"])
        needs_review = result_data.get("needs_review", False)
        
    except Exception as e:
        print(f"计划解析失败，降级处理: {e}")
        plan_steps = ["Research", "Write"]
        needs_review = True # 解析失败时，为了安全，默认开启审批
    
    # 初始步骤
    first_step = plan_steps[0] if plan_steps else "END"
    return {"plan": plan_steps, "current_step": first_step, "needs_review": needs_review}

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

      【代码执行与防幻觉协议】(CRITICAL):
      1. **Fact Check**: 如果你没有编写代码调用 `python_interpreter`，**绝对禁止**在回复中声称“文件已保存”或提及 `/app/output.png`。
      2. 只有当你生成的 Python 代码中确实包含 `plt.savefig('/app/output.png')` 时，才允许在代码注释或最终总结中提及该文件。
      3. 如果任务不需要代码（例如纯逻辑分析），请直接给出文字结论，不要假装运行了代码。
      
      【代码执行要求】:
      1. 务必使用 print() 输出最终结果。
      2. 仅在需要时绘图 (/app/output.png)。
      3. **不要**在代码中设置绘图风格 (如 `sns.set_theme`)，环境已预置了支持中文的学术风格配置。直接画图即可。
      4. 优先使用 **Seaborn** (`sns`) 绘图，其次是 Matplotlib (`plt`)。
      
      【通用美学与配色规范】(必须严格遵守):
      1. **严禁硬编码颜色**: 禁止出现 `color=['red', 'blue']` 或 `c='k'` 这种代码。环境已预置了优雅的 `custom_colors`，请利用它。
      
      2. **分类图表 (柱状图 bar / 散点图 scatter / 箱线图 box / 小提琴图 violin)**:
         - **规则**: 必须通过 `hue` 参数激活自动配色。
         - **写法**: `sns.barplot(x=vars, y=vals, hue=vars, legend=False)`
         - **原理**: 如果不加 `hue`，Seaborn 默认只用一种颜色；加上 `hue` 才会按类别循环使用全局色盘。
         
      3. **饼图 (Pie Chart)**:
         - **规则**: Matplotlib 的饼图不会自动通过 hue 取色，必须手动调用全局色盘。
         - **写法**: `plt.pie(values, labels=labels, colors=sns.color_palette(), autopct='%1.1f%%')`
         - **注意**: 必须传 `colors=sns.color_palette()`，否则它会变回丑陋的默认蓝/橙色。
         
      4. **折线图 (Line Chart)**:
         - **规则**: 多条线请用 `hue` 分组；单条线保持默认颜色即可（会自动使用色盘的第一个主色，非常协调）。
         
      5. **热力图 (Heatmap)**:
         - **规则**: 禁止使用默认配色。
         - **推荐**: `cmap='YlGnBu'` (清新蓝绿，适合一般数据) 或 `cmap='RdBu_r'` (红蓝发散，适合有正负对比的数据)。
         - **写法**: `sns.heatmap(data, cmap='YlGnBu', annot=True)`
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

    # 检查是否发生了“没调工具却声称有文件”的情况
    if not response.tool_calls and "/app/output.png" in str(analysis_output):
        print("--- [系统纠错] 检测到 Analyst 幻觉 (未执行代码却声称有图)，正在修正... ---")
        analysis_output = "[系统提示]: 分析师未执行任何代码，因此没有生成图表。请忽略关于 /app/output.png 的描述，仅参考文字分析。"
    
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

# 5. [新增] 人工审批节点 (HumanReview)
def human_review_node(state: AgentState) -> dict:
    print("--- [人工审批] 等待用户确认... ---")
    # 这里什么都不用做，只是为了让图停在这里
    return {}

# --- 5. 路由逻辑 (通用化) ---

def universal_router(state: AgentState) -> str:
    """通用路由器: 根据 current_step 直接映射到节点"""
    step = state.get("current_step")
    
    if step == "Research": return "Researcher"
    if step == "Analyze": return "Analyst"
    if step == "Write": return "Writer"
    if step == "END": return END
    return END # 默认

def planner_router(state: AgentState) -> str:
    """规划师路由: 决定是去审批，还是直接开始干活"""
    needs_review = state.get("needs_review", False)
    
    if needs_review:
        return "HumanReview" # 去审批节点（会被中断）
    else:
        # 不需要审批，直接去第一步 (复用通用路由逻辑)
        return universal_router(state)

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
workflow.add_node("HumanReview", human_review_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Planner")

# Planner -> Router
workflow.add_conditional_edges(
    "Planner",
    planner_router, # 使用新路由
    {
        "HumanReview": "HumanReview", 
        "Researcher": "Researcher",
        "Analyst": "Analyst", 
        "Writer": "Writer", 
        "END": END
    }
)

# [新增] HumanReview -> Universal Router
# 如果用户批准了（继续运行），就从这里进入第一步
workflow.add_conditional_edges("HumanReview", universal_router)

# Researcher -> Router
workflow.add_conditional_edges("Researcher", universal_router)

# Analyst -> QC Router -> (Analyst or Next Node)
workflow.add_conditional_edges("Analyst", qc_router)

# Writer -> End
workflow.add_conditional_edges("Writer", lambda x: END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["HumanReview"])