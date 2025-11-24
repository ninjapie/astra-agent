# tools.py
import docker
import tarfile
import io
import json
from langchain_core.tools import tool
import os
import uuid
import time
import requests
from bs4 import BeautifulSoup

try:
    client = docker.from_env()
except Exception as e:
    print(f"⚠️ 警告: 无法连接到 Docker。错误: {e}")
    client = None

# --- 1. 定义全局风格配置代码 ---
# 这段代码会被注入到所有 Python 脚本的开头
GLOBAL_STYLE_CONFIG = """
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
import subprocess
import sys

def install(package):
    print(f"--- [系统] 正在安装依赖: {package} ... ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"--- [系统] {package} 安装成功! ---")
    except Exception as e:
        print(f"--- [系统警告] {package} 安装失败: {e} ---")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

try:
    sns.set_theme(context="paper", style="whitegrid", palette="muted", font_scale=1.2)
    custom_colors = [
        "#2a9d8f", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
        "#76B7B2", "#EDC948", "#9C755F", "#FF9DA7", "#BAB0AC",
        "#A0CBE8", "#FFBE7D", "#8CD17D", "#FF9D9A", "#D4A6C8",
        "#499894", "#E19D29", "#79706E", "#FABFD2", "#86BCB6"
    ]
    sns.set_palette(custom_colors)
    base_fonts = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
    
    plt.rcParams['font.sans-serif'] = base_fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['font.serif'] = ['Times New Roman'] + base_fonts + plt.rcParams['font.serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
except Exception as e:
    print(f"[系统警告] 风格配置加载失败: {e}")
"""

@tool
def python_interpreter(code: str) -> str:
    """
    Python 代码执行沙箱 (支持绘图)。
    环境已预装: pandas, matplotlib, seaborn, numpy, sklearn。
    
    绘图指南:
    1. 不要使用 plt.show()。
    2. 必须将图表保存为 '/app/output.png'，例如: plt.savefig('/app/output.png')。
    3. 必须使用 print() 输出文本结果。
    """
    if not client:
        return json.dumps({"error": "Docker 服务未启动"})
    
    max_retries = 3
    retry_delay = 2 # 秒
    
    last_error = None
    container = None

    for attempt in range(max_retries):
        try:
            # 1. 启动容器 (使用我们预构建的镜像)
            container = client.containers.run(
                "astra-sandbox", # <--- 使用专用镜像
                detach=True,
                mem_limit="512m", # 绘图可能需要多一点内存
                cpu_period=100000,
                cpu_quota=50000,
            )

            # 2. 准备代码
            # 将我们的配置代码 + 用户的代码拼接在一起
            full_code = GLOBAL_STYLE_CONFIG + "\n\n" + code
            print(f"--- [沙箱] 尝试第 {attempt+1} 次执行... ---")

            script_content = full_code.encode('utf-8')
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar_info = tarfile.TarInfo(name='script.py')
                tar_info.size = len(script_content)
                tar_stream.seek(0)
                tar.addfile(tar_info, io.BytesIO(script_content))
            
            tar_stream.seek(0)
            
            # 3. 上传代码
            container.put_archive("/app", tar_stream)

            # 4. 执行代码
            exec_result = container.exec_run("python /app/script.py")
            logs = exec_result.output.decode("utf-8")

            # --- M7: 通用文件提取 (图片 + 文档) ---
            saved_files = []
            
            # 定义我们要从沙箱里"打捞"的文件扩展名
            target_extensions = ['.png', '.jpg', '.jpeg', '.xlsx', '.csv', '.pdf', '.txt']
            
            try:
                # 列出 /app 目录下的所有文件
                # (这是一个简单的 hack，通过 ls 命令查看生成了什么)
                ls_result = container.exec_run("ls /app")
                if ls_result.exit_code == 0:
                    file_list = ls_result.output.decode("utf-8").split()
                    
                    for filename in file_list:
                        # 检查是否是我们感兴趣的文件类型，且不是我们上传的 script.py
                        if any(filename.endswith(ext) for ext in target_extensions) and filename != 'script.py':
                            
                            # 从容器提取文件
                            bits, stat = container.get_archive(f"/app/{filename}")
                            
                            file_stream = io.BytesIO()
                            for chunk in bits:
                                file_stream.write(chunk)
                            file_stream.seek(0)
                            
                            with tarfile.open(fileobj=file_stream) as tar:
                                extract_file = tar.extractfile(filename)
                                if extract_file:
                                    file_bytes = extract_file.read()
                                    
                                    # (关键) 生成一个唯一的服务器端文件名，防止冲突
                                    unique_filename = f"{uuid.uuid4()}_{filename}"
                                    save_path = os.path.join("static", unique_filename)
                                    
                                    # 保存到宿主机的 static 目录
                                    with open(save_path, "wb") as f:
                                        f.write(file_bytes)
                                        
                                    # 记录文件信息
                                    # --- M7 修复: 智能区分图片和下载文件 ---
                                    # 1. 如果是图片，继续用 /static/ (为了在前端 <img> 标签里直接显示)
                                    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                                        file_url = f"http://127.0.0.1:8000/static/{unique_filename}"
                                        file_type = "image"
                                    # 2. 如果是其他文件 (Excel/PDF)，改用 /download/ (为了正确下载文件名)
                                    else:
                                        file_url = f"http://127.0.0.1:8000/download/{unique_filename}"
                                        file_type = "file"

                                    saved_files.append({
                                        "original_name": filename,
                                        "url": file_url,
                                        "type": file_type,
                                        "saved_path": save_path # [M12 新增] 返回本地路径，供视觉模型读取
                                    })
                                    print(f"--- [沙箱] 成功提取文件: {filename} -> {unique_filename} ---")

            except Exception as e:
                print(f"文件提取出错: {e}")

            # 6. 构造返回
            result = {
                "stdout": logs,
                "files": saved_files, # 返回文件列表
                "exit_code": exec_result.exit_code
            }
            
            return json.dumps(result)

        except Exception as e:
            print(f"⚠️ [沙箱] 第 {attempt+1} 次失败: {e}")
            last_error = e
            time.sleep(retry_delay) # 等待后重试
            
        finally:
            if container:
                try:
                    container.remove(force=True) # 强制删除，这会同时停止容器
                except Exception as cleanup_error:
                    print(f"⚠️ [沙箱] 容器清理失败: {cleanup_error}")
      # 如果重试都失败了
    return json.dumps({"error": f"Docker 执行连续失败 (Max Retries Exceeded): {last_error}"})

# 网页抓取工具
@tool
def scrape_website(url: str) -> str:
    """
    访问并抓取指定 URL 的网页正文内容。
    适用于获取长文章、新闻报道或技术文档的详细内容。
    """
    print(f"--- [浏览器] 正在访问: {url} ---")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding # 自动识别编码
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除无关元素 (脚本、样式、导航等)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # 提取文本
        text = soup.get_text(separator="\n")
        
        # 清理空行
        clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
        # 截断过长内容 (防止爆 Token)
        if len(clean_text) > 8000:
            clean_text = clean_text[:8000] + "\n...(内容过长已截断)"
            
        return clean_text

    except Exception as e:
        return f"网页抓取失败: {str(e)}"