# tools.py
import docker
import tarfile
import io
import json
from langchain_core.tools import tool
import os
import uuid # 新增: 用于生成唯一文件名

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

# [Astra 修复] 忽略 Seaborn 的版本兼容性警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# [Astra 全局绘图风格配置]
try:
    # 1. 设置 Seaborn 样式 (这会重置字体，所以必须先运行)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    
    # 2. [关键修复] 定义字体回退列表 (Font Fallback)
    # 只要列表里有一个能用的中文字体，Matplotlib 就能正常显示中文
    # 我们在 Docker 里装了 'Noto Sans CJK JP' (思源黑体)
    base_fonts = ['Noto Sans CJK JP', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
    
    # 3. 暴力覆盖: 无论是用 serif 还是 sans-serif，都强制包含中文字体
    # 更新无衬线字体栈 (Seaborn 默认用这个)
    plt.rcParams['font.sans-serif'] = base_fonts + plt.rcParams['font.sans-serif']
    
    # 更新衬线字体栈 (以防用户手动指定 serif)
    plt.rcParams['font.serif'] = ['Times New Roman'] + base_fonts + plt.rcParams['font.serif']
    
    # 4. 设置默认字体族
    # 解释: 虽然学术界喜欢 Serif (Times New Roman)，但图表通常用 Sans-Serif 更清晰
    # 且 'Noto Sans CJK SC' 是无衬线体，匹配 sans-serif 效果最好
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 5. 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 6. 高清设置
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

    # print(f"--- [沙箱] 准备执行代码... ---\n{code}...")

    container = None
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
        print(f"--- [沙箱] 准备执行代码... ---\n{full_code[:50]}")
        script_content = full_code.encode('utf-8')
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w') as tar:
            tar_info = tarfile.TarInfo(name='script.py')
            tar_info.size = len(script_content)
            tar_stream.seek(0)
            tar.addfile(tar_info, io.BytesIO(script_content))
        
        tar_stream.seek(0)
        
        # 3. 上传代码
        # (因为 Dockerfile 已经定义了 WORKDIR /app，所以目录肯定存在)
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
                                    "type": file_type
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
        return json.dumps({"error": str(e)})
        
    finally:
        if container:
            try:
                container.stop()
                container.remove()
            except:
                pass