# 使用官方 Python 3.11 轻量版
FROM python:3.11-slim

# 防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. 准备安装字体 (需要开启 contrib 和 non-free 源)
#    Debian 默认只开启 main，没有微软字体
RUN echo "deb http://deb.debian.org/debian bookworm contrib non-free" > /etc/apt/sources.list.d/contrib.list

# 2. 安装系统依赖 + 微软字体
#    ttf-mscorefonts-installer 安装时会弹出许可协议，我们需要用 debconf 自动同意
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    fontconfig \
    debconf && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y --no-install-recommends ttf-mscorefonts-installer && \
    fc-cache -f -v && \
    rm -rf /var/lib/apt/lists/*

# 3. 预装数据科学全家桶
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn \
    numpy \
    scikit-learn \
    openpyxl \
    networkx \
    scipy

# 创建工作目录
WORKDIR /app

# 默认命令
CMD ["sleep", "infinity"]