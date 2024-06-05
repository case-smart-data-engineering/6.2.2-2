# 基础镜像
FROM gitpod/workspace-full

# 安装 Python 3.8 和相关工具
RUN sudo apt-get update && \
    sudo apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip && \
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# 设置默认的 pip3
RUN python3.8 -m pip install --upgrade pip
