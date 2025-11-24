import os
import operator
import base64
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
from tools import python_interpreter, scrape_website

# --- 1. 配置与初始化 ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)
web_search_tool = TavilySearch(max_results=3)

DB_NAME = "astra_db"
DB_USER = "maple" 
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

# --- [修复 1] 定义 Reducer 函数，允许最新值覆盖旧值 ---
def replace_str(current: str, new: str) -> str:
    return new

# --- 2. 状态定义 ---
class AgentState(TypedDict):
    task: str
    plan: List[str] 
    final_report: str
    current_step: str
    analysis_results: Annotated[List[str], operator.add]
    retry_count: int
    image_data: str
    needs_review: bool
    visual_critique: Annotated[str, replace_str] 
    latest_image_path: str 

    # [优雅重构] 删除 last_tool_used，增加通用行为标记
    did_call_tool: bool  # 这轮是否动手了？
    has_generated_image: bool # 这轮是否出图了？
    has_generated_html: bool # [M16 新增] 标记是否生成了 HTML

# --- 3. 辅助函数 ---
def get_next_step_name(plan: List[str], current_step_name: str) -> str:
    try:
        current_index = plan.index(current_step_name)
        if current_index + 1 < len(plan):
            return plan[current_index + 1]
        else:
            return "END"
    except ValueError:
        return "END"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- 4. 节点定义 ---

# 1. 规划师 (Planner)
async def planner_node(state: AgentState) -> dict:
    print("--- [规划师] 开始工作 ---")
    task = state.get("task")
    
    prompt = f"""
    你是一个专业的项目规划师。任务: {task}
    请制定步骤计划，从以下选择:
    1. "Research": 需要外部信息。
    2. "Analyze": 需要计算、代码执行、浏览网页或生成文件。
    3. "Write": 需要写一份详细的文字总结报告或分析内容。
    
    【关键规则】:
    - 如果任务只是要求直接的答案、计算结果或生成特定文件(如"生成5个名字", "画张图")，**不要**包含 "Write"。

    同时，请判断该任务是否需要【用户审批】(needs_review):
    - 如果任务有风险（如删除文件）、非常复杂、或者指令模糊不确定，请设为 true。
    - 如果任务很简单、明确，请设为 false (自动执行)。
    
    【重要】请严格返回以下 JSON 格式（纯 JSON）:
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
        clean_str = full_plan_str.replace("```json", "").replace("```", "").strip()
        result_data = json.loads(clean_str)
        plan_steps = result_data.get("plan", ["Research", "Write"])
        needs_review = result_data.get("needs_review", False)
    except Exception as e:
        print(f"计划解析失败，降级处理: {e}")
        plan_steps = ["Research", "Write"]
        needs_review = True 
    
    first_step = plan_steps[0] if plan_steps else "END"
    return {"plan": plan_steps, "current_step": first_step, "needs_review": needs_review}

# 2. 研究员 (Researcher)
def researcher_node(state: AgentState) -> dict:
    print("--- [研究员] 开始搜索 ---")
    task = state.get("task")
    print(f'task: {task}')
    try:
        research_results = web_search_tool.invoke(task)
    except Exception as e:
        research_results = [f"搜索失败: {e}"]
    
    print(f"search results: {research_results}")
    MAX_CHARS = 4000
    documents = []
    for result in research_results['results']:
        content = str(result)
        if len(content) > MAX_CHARS: content = content[:MAX_CHARS] + "..."
        documents.append(Document(text=content, metadata={"task": task}))
    
    if documents:
        index.insert_nodes(documents)
    
    plan = state.get("plan", [])
    next_step = get_next_step_name(plan, "Research")
    print(f"next step: {next_step}")
    return {"current_step": next_step}

# 3. 分析师 (Analyst)
async def analyst_node(state: AgentState) -> dict:
    print("--- [分析师] 开始分析 ---")
    task = state.get("task")
    image_data = state.get("image_data") 
    previous_results = state.get("analysis_results", [])
    retry_count = state.get("retry_count", 0)
    visual_critique = state.get("visual_critique", "PASS") 
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    rag_context = await query_engine.aquery(task)

    # [M15 核心] 构建完整的历史记忆上下文 (Memory Stream)
    history_context = ""
    if previous_results:
        history_context = "\n\n=== 📜 历史操作记录 (Memory Stream) ===\n"
        for i, res in enumerate(previous_results):
            # 截断过长的输出 (如网页全文)，但保留足够长度供分析
            preview = res[:3000] + "...(内容过长已截断)" if len(res) > 3000 else res
            history_context += f"--- Step {i+1} Output ---\n{preview}\n\n"
        history_context += "========================================\n"
    
    prompt = f"""
      你是一个全能数据分析师。任务: {task}
      背景: {rag_context}

      {history_context}

      【当前状态与决策】:
      请回顾上面的 [历史操作记录] 来决定下一步：
      1. **信息不足？** -> 调用 `scrape_website(url)` 获取详情。
      2. **缺库？** -> 调用 `install('package')`。
      3. **有数据了？** -> 编写 Python 代码处理数据或画图。
      4. **报错了？** -> 根据错误信息修正代码。

      【交互式图表 (Interactive)】:
      1. 如果用户要求“交互式”、“动态”或“可缩放”的图表，**必须**生成 HTML 文件。
      2. **推荐库**: `pyecharts` (首选) 或 `plotly`。
      3. **安装**: 别忘了先 `install('pyecharts')`。
      4. **保存**: 必须渲染并保存为 `/app/output.html` (例如 `bar.render("/app/output.html")`)。

      【深度浏览】:
      1. 如果背景信息(rag_context)太简略，或者包含 URL 链接，你可以使用 `scrape_website(url)` 工具来读取网页全文。
      2. 这是一个获取详细技术文档、长篇报道或具体参数的绝佳方式。
      3. **流程建议**: 觉得信息不够 -> `scrape_website` -> 获得信息 -> `python_interpreter` 处理。

      【动态依赖管理】(IMPORTANT):
      1. 环境已内置 `install(package_name)` 函数。
      2. 如果任务需要外部库（如 `wordcloud`, `qrcode`, `yfinance`, `openpyxl` 等），**必须**在 import 之前调用它。
      3. **示例**:
         ```python
         install("wordcloud") # 先安装
         from wordcloud import WordCloud # 后导入
         # ... 你的代码
         ```
    
      【反死循环协议】(CRITICAL):
      1. **严禁重复操作**: 在调用工具前，必须检查 [历史操作记录]。如果你（或之前的步骤）已经抓取过某个 URL，**绝不允许**再次抓取同一个 URL。
      2. **处理失败**: 如果上一步抓取的内容不理想（如为空或乱码），请尝试更换 URL，或者直接基于现有信息编写代码，**不要**原地重试。
      3. **推进流程**: 如果你已经有了抓取结果（即使只是部分），请立即转入下一个步骤。

      【代码执行与防幻觉协议】(CRITICAL):
      1. **Fact Check**: 如果你没有编写代码调用 `python_interpreter`，**绝对禁止**在回复中声称“文件已保存”或提及 `/app/output.png`。
      2. 只有当你生成的 Python 代码中确实包含 `plt.savefig('/app/output.png')` 时，才允许在代码注释或最终总结中提及该文件。
      3. 如果任务不需要代码，请直接给出文字结论，不要假装运行了代码。
      
      【代码执行要求】:
      1. 务必使用 print() 输出最终结果。
      2. 仅在需要时绘图 (/app/output.png)。
      3. **不要**在代码中设置绘图风格，环境已预置。
      4. 优先使用 **Seaborn** (`sns`) 绘图。
      
      【通用美学与配色规范】(必须严格遵守):
      1. **严禁硬编码颜色**: 禁止出现 `color=['red', 'blue']`。
      2. **分类图表**: 必须通过 `hue` 参数激活自动配色，例如 `sns.barplot(x=vars, y=vals, hue=vars, legend=False)`。
      3. **饼图**: 必须手动调用色盘 `plt.pie(..., colors=sns.color_palette())`。
      4. **热力图**: 推荐 `cmap='YlGnBu'`。
    """
    
    # 视觉修正 Prompt
    if visual_critique and visual_critique != "PASS":
        prompt += f"\n\n🔥🔥🔥【视觉修正模式】🔥🔥🔥\n上一版代码生成的图片被视觉模型驳回。意见如下:\n{visual_critique}\n请修改代码以修复上述审美或显示问题。"
    elif retry_count > 0 and previous_results:
         prompt += f"\n【修复错误】上次失败: {previous_results[-1]}\n请修正代码。"
    
    if image_data:
        print("--- [分析师] 收到图片, 正在进行视觉分析... ---")
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    else:
        message_content = prompt
    
    analyst_llm = llm.bind_tools([python_interpreter, scrape_website])
    messages = [HumanMessage(content=message_content)]
    
    response = await analyst_llm.ainvoke(messages)
    analysis_output = response.content

    did_call_tool = False
    has_generated_image = False
    has_generated_html = False
    latest_image_path = None

    if response.tool_calls:
        did_call_tool = True
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "python_interpreter":
                code = tool_args["code"]
                print(f"--- [分析师] 执行代码... ---")
                tool_result = python_interpreter.invoke(code)
                analysis_output = tool_result

                try:
                    output_data = json.loads(tool_result)
                    files = output_data.get("files", [])
                    for f in files:
                        if f.get("type") == "html":
                            has_generated_html = True
                            print(f"--- [分析师] 生成了交互式 HTML ---")
                        elif f.get("type") == "image" and f.get("saved_path"):
                            latest_image_path = f["saved_path"]
                            has_generated_image = True
                            print(f"--- [分析师] 捕获到生成图片: {latest_image_path} ---")
                except:
                    pass
            elif tool_name == "scrape_website":
                print(f"--- [分析师] 调用浏览器抓取: {tool_args.get('url')} ---")
                tool_result = scrape_website.invoke(tool_args.get("url"))
                # 将抓取结果作为补充信息，可能需要再次思考?
                # 简化起见，我们将结果存入 analysis_output，供下一轮或 Writer 使用
                analysis_output = f"【网页抓取结果】:\n{tool_result[:2000]}..." # 预览
    # 防幻觉检查
    if not did_call_tool and "/app/output.png" in str(analysis_output):
        print("--- [系统纠错] 检测到 Analyst 幻觉，正在修正... ---")
        warning_msg = "[系统提示]: 分析师未执行任何代码，因此没有生成图表。请忽略关于 /app/output.png 的描述，仅参考文字分析。"
        analysis_output += warning_msg

    current_retry = state.get("retry_count", 0)
    
    return {
        "analysis_results": [analysis_output], 
        "retry_count": current_retry + 1,
        "latest_image_path": latest_image_path,
        "did_call_tool": did_call_tool,          # [新]
        "has_generated_image": has_generated_image, # [新]
        "has_generated_html": has_generated_html
    }

# 4. 视觉评论家 (Visual Critic)
async def visual_critic_node(state: AgentState) -> dict:
    # [M16] 如果是 HTML，目前 Visual Critic 无法检查 (因为是代码)，直接通过
    # 未来可以加入代码检查，但现在先放行
    if state.get("has_generated_html"):
        print("--- [视觉评论家] 检测到 HTML 交互图表，自动通过 (HTML Bypass) ---")
        return {"visual_critique": "PASS"}

    image_path = state.get("latest_image_path")
    
    # 无图则直接通过
    if not image_path or not os.path.exists(image_path):
        print("--- [视觉评论家] 无图片，跳过检查 ---")
        return {"visual_critique": "PASS"}
    
    print(f"--- [视觉评论家] 正在审查图片: {image_path} ---")
    base64_image = encode_image(image_path)

    # [修复] 获取更丰富的上下文
    task_description = state.get("task", "未提供任务描述")
    analysis_results = state.get("analysis_results", [])

    # 尝试获取最近的工具输出作为参考，但主要依靠 task
    tool_output_snippet = analysis_results[-1][:500] if analysis_results else "无工具输出"
    
    critic_prompt = f"""
    你是一个通用的视觉质量控制专家 (Visual QA)。

    【原始任务目标】:
    "{task_description}"
    
    【最近的执行日志(仅供参考)】:
    {tool_output_snippet}...
    
    【任务】:
    请检查上传的图片是否符合【原始任务目标】，并检查是否存在**明显的视觉缺陷**。
    
    【判断标准】:
    1. **相关性**: 图片内容是否看似在响应用户的任务？(例如用户要二维码，图就是二维码)。
       - 注意：如果执行日志是 JSON 格式或代码日志，请忽略日志内容，**重点对比图片和[原始任务目标]**。
    2. **可读性**: 是否存在乱码、严重模糊、内容被截断？
    3. **完整性**: 图片是否完整？
    
    如果不符合描述或有严重缺陷，请给出具体的修改建议。
    如果图片符合描述且质量合格，或者你不确定但图片看起来没有技术错误，请**仅回复 "PASS"**。
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": critic_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = await llm.ainvoke([message])
    critique = response.content
    print(f"--- [视觉评论家] 意见: {critique} ---")
    
    return {"visual_critique": critique}

# 5. 写手 (Writer)
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
    请撰写报告。如果生成了 HTML 图表，请在报告中提示用户下载或查看附件。
    """
    messages = [HumanMessage(content=prompt)]
    
    full_report = ""
    async for chunk in llm.astream(messages):
        full_report += chunk.content or ""

    return {"final_report": full_report, "current_step": "END"}

# 6. 人工审批节点
def human_review_node(state: AgentState) -> dict:
    print("--- [人工审批] 等待用户确认... ---")
    return {}

# --- 5. 路由逻辑 ---

def universal_router(state: AgentState) -> str:
    step = state.get("current_step")
    if step == "Research": return "Researcher"
    if step == "Analyze": return "Analyst"
    if step == "Write": return "Writer"
    if step == "END": return "END"
    return "END"

def planner_router(state: AgentState) -> str:
    if state.get("needs_review"): return "HumanReview"
    return universal_router(state)

def analyst_router(state: AgentState) -> str:
    """
    通用行为路由：
    1. 没调工具? -> 说明想明白了/没事干了 -> 去 Writer
    2. 调了工具且出图了? -> 说明有作品 -> 去 VisualCritic 检查
    3. 调了工具但没出图? -> 说明只是由获取了中间信息(抓取/计算) -> 回 Analyst 继续消化
    """
    did_call_tool = state.get("did_call_tool", False)
    has_image = state.get("has_generated_image", False)
    has_html = state.get("has_generated_html", False)
    retry_count = state.get("retry_count", 0)
    
    # 1. [思考结束]：没动手，只是在说话 -> 任务完成
    if not did_call_tool:
        print("--- [路由] 分析师未调用工具，认为分析结束 ---")
        
        # 还要检查一下有没有报错 (兼容旧的 QC 逻辑)
        # ... (可以保留简单的文本错误检查，也可以省略)
        
        plan = state.get("plan", [])
        next_step = get_next_step_name(plan, "Analyze")
        if next_step == "Write": return "Writer"
        return "END"

    # 2. [作品产出]：动手了，且生成了图片 -> 去质检
    if has_image or has_html:
        print("--- [路由] 检测到视觉产出 (Img/HTML) -> VisualCritic ---")
        return "VisualCritic"

    # 3. [中间状态]：动手了(比如抓取/安装)，但没出图 -> 必定是中间步骤
    # 强制闭环，让 Analyst 消化刚才获得的信息
    if did_call_tool:
        if retry_count < 15: 
            print("--- [路由] 工具执行完毕(无图) -> Analyst 继续思考 ---")
            return "Analyst"
        else:
            return "Writer"
    
    print("--- [路由] 分析师停止操作 -> Writer ---")
    plan = state.get("plan", [])
    next_step = get_next_step_name(plan, "Analyze")
    if next_step == "Write": return "Writer"
    return "END"

# [修复 5] 视觉路由：通过后去 Writer
def critic_router(state: AgentState) -> str:
    critique = state.get("visual_critique", "PASS")
    retry_count = state.get("retry_count", 0)
    
    # 1. 通过，去 Writer (或者根据 plan 决定)
    if critique == "PASS":
        plan = state.get("plan", [])
        next_step = get_next_step_name(plan, "Analyze")
        if next_step == "Write": 
            return "Writer"
        else:
            print("--- [流程] 计划已完成，任务结束 ---")
            return "END"
        
    # 2. 不通过，回炉重造
    if retry_count < 3:
        print("🔙 [视觉路由] 驳回! 返回 Analyst 重画...")
        return "Analyst"
    
    # 3. 实在改不动了，去 Writer
    return "Writer"

# --- 6. 构建图 ---
workflow = StateGraph(AgentState)

workflow.add_node("Planner", planner_node)
workflow.add_node("HumanReview", human_review_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("VisualCritic", visual_critic_node)
workflow.add_node("Writer", writer_node)

workflow.set_entry_point("Planner")

# Planner 路由
workflow.add_conditional_edges(
    "Planner",
    planner_router,
    {"HumanReview": "HumanReview", "Researcher": "Researcher", "Analyst": "Analyst", "Writer": "Writer", "END": END}
)

workflow.add_conditional_edges(
    "HumanReview",
    universal_router,
    {"Researcher": "Researcher", "Analyst": "Analyst", "Writer": "Writer", "END": END}
)
workflow.add_conditional_edges(
    "Researcher",
    universal_router,
    {"Researcher": "Researcher", "Analyst": "Analyst", "Writer": "Writer", "END": END}
)

# [修复 6] Analyst 只有一条条件出边
workflow.add_conditional_edges(
    "Analyst",
    analyst_router,
    {"Analyst": "Analyst", "VisualCritic": "VisualCritic", "Writer": "Writer", "END": END}
)

# VisualCritic 路由
workflow.add_conditional_edges(
    "VisualCritic",
    critic_router,
    {"Analyst": "Analyst", "Writer": "Writer", "END": END}
)

workflow.add_conditional_edges("Writer", lambda x: END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["HumanReview"])