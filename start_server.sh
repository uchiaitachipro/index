#!/bin/bash

# IndexTTS WebUI 启动脚本
# 功能: 检测并清理 6006 端口占用,然后启动 api_server.py

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PORT=6006
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="py312"

# 默认启动参数 (可通过环境变量覆盖)
USE_DEEPSPEED="${USE_DEEPSPEED:-false}"      # 默认禁用 DeepSpeed (需要 g++ 编译器)
USE_FP16="${USE_FP16:-false}"                # 默认不启用 FP16
USE_CUDA_KERNEL="${USE_CUDA_KERNEL:-false}"  # 默认不启用 CUDA kernel
USE_VERBOSE="${USE_VERBOSE:-false}"          # 默认不启用详细日志

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查端口占用并终止进程
kill_process_on_port() {
    local port=$1

    log_info "检查端口 ${port} 的占用情况..."

    # 使用 lsof 查找占用端口的进程 (更可靠)
    if command -v lsof &> /dev/null; then
        # 获取所有占用该端口的进程 PID
        local pids=$(lsof -ti :${port} 2>/dev/null || true)

        if [ -z "$pids" ]; then
            log_info "端口 ${port} 未被占用"
            return 0
        fi

        log_warn "发现端口 ${port} 被以下进程占用: $pids"

        # 遍历每个 PID
        for pid in $pids; do
            # 获取进程信息
            local proc_info=$(ps -p $pid -o pid,comm,args 2>/dev/null || echo "进程已不存在")
            log_warn "进程信息:\n$proc_info"

            # 先尝试优雅终止 (SIGTERM)
            log_info "尝试优雅终止进程 $pid (SIGTERM)..."
            kill -15 $pid 2>/dev/null || true

            # 等待进程退出 (最多 5 秒)
            local count=0
            while [ $count -lt 10 ]; do
                if ! kill -0 $pid 2>/dev/null; then
                    log_info "进程 $pid 已优雅退出"
                    break
                fi
                sleep 0.5
                count=$((count + 1))
            done

            # 如果进程仍然存在,强制终止 (SIGKILL)
            if kill -0 $pid 2>/dev/null; then
                log_warn "进程 $pid 未响应 SIGTERM,执行强制终止 (SIGKILL)..."
                kill -9 $pid 2>/dev/null || true
                sleep 1

                if kill -0 $pid 2>/dev/null; then
                    log_error "无法终止进程 $pid"
                    return 1
                else
                    log_info "进程 $pid 已被强制终止"
                fi
            fi
        done

    # 备用方案: 使用 fuser
    elif command -v fuser &> /dev/null; then
        log_info "使用 fuser 检查端口..."
        local pids=$(fuser ${port}/tcp 2>/dev/null | tr -d ' ' || true)

        if [ -z "$pids" ]; then
            log_info "端口 ${port} 未被占用"
            return 0
        fi

        log_warn "使用 fuser 终止端口 ${port} 上的进程..."
        fuser -k -15 ${port}/tcp 2>/dev/null || true
        sleep 2

        # 检查是否还有进程占用
        if fuser ${port}/tcp 2>/dev/null; then
            log_warn "执行强制终止..."
            fuser -k -9 ${port}/tcp 2>/dev/null || true
        fi

    else
        log_error "未找到 lsof 或 fuser 命令,无法检查端口占用"
        log_error "请手动安装: apt-get install lsof 或 apt-get install psmisc"
        return 1
    fi

    # 最终验证端口是否已释放
    sleep 1
    if lsof -ti :${port} &>/dev/null || fuser ${port}/tcp &>/dev/null 2>&1; then
        log_error "端口 ${port} 仍被占用,启动可能失败"
        return 1
    else
        log_info "端口 ${port} 已成功释放"
    fi
}

# 激活 conda 环境
activate_conda() {
    log_info "激活 conda 环境: ${CONDA_ENV}"

    # 初始化 conda (如果需要)
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        log_error "未找到 conda,请确保已安装 Anaconda 或 Miniconda"
        exit 1
    fi

    # 激活环境
    conda activate ${CONDA_ENV} || {
        log_error "无法激活 conda 环境 '${CONDA_ENV}'"
        log_error "请先创建环境: conda create -n ${CONDA_ENV} python=3.12"
        exit 1
    }

    # 设置语言环境为中文 (强制 Gradio 使用中文)
    export LANG=zh_CN.UTF-8
    export LC_ALL=zh_CN.UTF-8
    export LANGUAGE=zh_CN:zh

    log_info "当前 Python: $(which python)"
    log_info "当前环境: ${CONDA_ENV}"
    log_info "语言设置: zh_CN.UTF-8"
}

# 检查必要文件
check_requirements() {
    log_info "检查必要文件..."

    if [ ! -f "${SCRIPT_DIR}/indextts/infer_v2_subtitle.py" ]; then
        log_error "未找到 infer_v2_subtitle.py,请确保在项目根目录下运行"
        exit 1
    fi

    if [ ! -f "${SCRIPT_DIR}/api_server.py" ]; then
        log_error "未找到 webui.py,请确保在项目根目录下运行"
        exit 1
    fi

    if [ ! -d "${SCRIPT_DIR}/checkpoints" ]; then
        log_warn "未找到 checkpoints 目录,首次运行可能需要下载模型"
    fi

    log_info "文件检查完成"
}

# 构建启动参数
build_launch_args() {
    local args=()

    # 用户传递的参数优先
    # 如果用户已指定 --deepspeed 或 --no-deepspeed,则尊重用户选择
    local user_specified_deepspeed=false
    local user_specified_fp16=false
    local user_specified_cuda_kernel=false
    local user_specified_verbose=false

    for arg in "$@"; do
        if [[ "$arg" == "--deepspeed" || "$arg" == "--no-deepspeed" ]]; then
            user_specified_deepspeed=true
        fi
        if [[ "$arg" == "--fp16" || "$arg" == "--no-fp16" ]]; then
            user_specified_fp16=true
        fi
        if [[ "$arg" == "--cuda_kernel" || "$arg" == "--no-cuda_kernel" ]]; then
            user_specified_cuda_kernel=true
        fi
        if [[ "$arg" == "--verbose" || "$arg" == "--no-verbose" ]]; then
            user_specified_verbose=true
        fi
    done

    # 如果用户没有指定,使用默认值
    if [ "$user_specified_deepspeed" = false ] && [ "$USE_DEEPSPEED" = "true" ]; then
        args+=("--deepspeed")
        # 输出到 stderr,避免污染返回值
        log_info "✓ 启用 DeepSpeed 加速 (默认)" >&2
    fi

    if [ "$user_specified_fp16" = false ] && [ "$USE_FP16" = "true" ]; then
        args+=("--fp16")
        log_info "✓ 启用 FP16 精度 (默认)" >&2
    fi

    if [ "$user_specified_cuda_kernel" = false ] && [ "$USE_CUDA_KERNEL" = "true" ]; then
        args+=("--cuda_kernel")
        log_info "✓ 启用 CUDA kernel (默认)" >&2
    fi

    if [ "$user_specified_verbose" = false ] && [ "$verbose" = "true" ]; then
        args+=("--verbose")
        log_info "✓ 启用详细日志 (默认)" >&2
    fi

    # 添加用户传递的所有参数
    args+=("$@")

    # 只输出参数数组到 stdout
    echo "${args[@]}"
}

# 主函数
main() {
    log_info "======================================"
    log_info "IndexTTS api_server 启动脚本"
    log_info "======================================"

    # 切换到脚本所在目录
    cd "${SCRIPT_DIR}"

    # 1. 检查必要文件
    check_requirements

    # 2. 清理端口占用
    kill_process_on_port ${PORT}

    # 3. 激活 conda 环境
    activate_conda

    # 4. 构建启动参数
    local launch_args=$(build_launch_args "$@")

    # 5. 启动 api_server
    log_info "======================================"
    log_info "启动 api_server.py (端口: ${PORT})..."
    log_info "======================================"

    # 使用 uv run 或直接运行 python
    if command -v uv &> /dev/null; then
        log_info "使用 uv run 启动..."
        uv run api_server.py ${launch_args}
    else
        log_info "使用 python 启动..."
        python api_server.py ${launch_args}
    fi
}

# 捕获 Ctrl+C 信号,优雅退出
trap 'log_warn "收到中断信号,正在退出..."; exit 130' INT TERM

# 执行主函数
main "$@"