#!/bin/bash

# ====================== 新增：信号处理与清理函数 ======================
# 定义清理函数：终止所有子进程并退出
cleanup() {
    echo -e "\n\n收到终止信号，正在停止所有正在执行的脚本..." >&2
    # 杀死所有父进程为当前脚本的子进程（即正在执行的sh脚本）
    # -P $$：匹配父进程ID为当前脚本PID的进程
    # 2>/dev/null：忽略“没有进程可杀死”的错误提示
    pkill -P $$ 2>/dev/null
    echo "所有子脚本已终止，主脚本退出。" >&2
    exit 1
}

# 捕获关键终止信号，触发清理函数
# SIGINT (2)：用户按Ctrl+C触发
# SIGTERM (15)：kill命令默认发送的信号
trap cleanup SIGINT SIGTERM
# =====================================================================

# 检查是否提供了目录参数，若未提供则使用当前目录
target_dir="${1:-.}"

# 确认目录是否存在
if [ ! -d "$target_dir" ]; then
    echo "错误：目录 '$target_dir' 不存在。" >&2
    exit 1
fi

# 获取目录中的所有sh文件
sh_files=("$target_dir"/*.sh)

# 检查是否有sh文件
if [ ${#sh_files[@]} -eq 0 ]; then
    echo "在目录 '$target_dir' 中未找到.sh文件。"
    exit 0
fi

# 遍历并执行sh文件
for file in "${sh_files[@]}"; do
    # 检查文件是否存在（处理通配符未匹配的情况）
    if [ ! -f "$file" ]; then
        continue
    fi

    echo "正在执行文件: $file"
    echo "----------------------------------------"

    # 执行脚本并捕获输出和状态
    if bash "$file"; then
        echo "✓ 执行成功: $file"
    else
        echo "✗ 执行失败: $file"
    fi

    echo "----------------------------------------"
    echo
done

echo "所有sh文件执行完毕！"