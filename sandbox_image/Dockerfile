# 使用官方 Python 3.11 轻量版
FROM python:3.11-slim

# 防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. 准备安装字体 (开启 contrib non-free)
RUN echo "deb http://deb.debian.org/debian bookworm contrib non-free" > /etc/apt/sources.list.d/contrib.list

# 2. 安装系统依赖 + 微软字体 + [新增] 中文字体
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    fontconfig \
    debconf \
    # [新增] 安装思源黑体 (支持中文/日文/韩文)
    fonts-noto-cjk \
    && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y --no-install-recommends ttf-mscorefonts-installer && \
    # 刷新字体缓存
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

WORKDIR /app
CMD ["sleep", "infinity"]