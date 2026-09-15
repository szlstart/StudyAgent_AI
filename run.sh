#!/bin/bash
# StudyAgent AI 快捷运行脚本（固定使用本机的 studybuddy Conda 环境）
# 用法：
#   ./run.sh                                      # 启动 Web 服务
#   ./run.sh data/test_data/run_test.py          # 完整测试
#   ./run.sh src/agents/homework/ocr_agent.py /tmp/hw.jpg math   # OCR 调试

CONDA_PYTHON="/opt/miniconda3/envs/studybuddy/bin/python"
SCRIPT="${1:-src/api/run_server.py}"

if [ ! -x "$CONDA_PYTHON" ]; then
    echo "未找到 Conda 环境: $CONDA_PYTHON" >&2
    echo "请先执行: conda create -p /opt/miniconda3/envs/studybuddy python=3.13" >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    shift
fi

cd "$(dirname "$0")"
exec "$CONDA_PYTHON" "$SCRIPT" "$@"
