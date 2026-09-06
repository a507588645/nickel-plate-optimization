FROM python:3.7-slim

# 设置工作目录
WORKDIR /app

# 复制必要的文件
COPY . /app/

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装依赖
COPY requirements.txt /app/
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 5005

# 启动命令
CMD ["python", "server.py"]