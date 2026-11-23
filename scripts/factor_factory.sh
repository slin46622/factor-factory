#!/usr/bin/env bash
# 因子工厂控制脚本
# 用法: ./scripts/factor_factory.sh <start|stop|pause|resume|status|dashboard|direction>

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # repo 根（pip 安装后可改 python -m factor_factory.daemon）
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
DAEMON="$PROJECT_DIR/scripts/factor_factory_daemon.py"
DASHBOARD="$PROJECT_DIR/scripts/factor_dashboard.py"
QLIB_DIR="${FF_HOME:-$HOME/.factor_factory}"
STOP_FILE="$QLIB_DIR/factor_factory.stop"
PAUSE_FILE="$QLIB_DIR/factor_factory.pause"
DIRECTION_FILE="$QLIB_DIR/factor_factory.direction.txt"
PID_FILE="$QLIB_DIR/factor_factory.pid"

mkdir -p "$QLIB_DIR"

case "${1:-status}" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "⚠️  因子工厂已在运行中 (PID $(cat "$PID_FILE"))"
            echo "   先执行 ./scripts/factor_factory.sh stop 再启动"
            exit 1
        fi
        # 清理历史停止信号
        rm -f "$STOP_FILE"
        echo "🚀 启动因子工厂..."
        nohup "$VENV_PYTHON" "$DAEMON" > "$QLIB_DIR/factor_factory.log" 2>&1 &
        echo $! > "$PID_FILE"
        echo "   PID: $!"
        echo "   日志: $QLIB_DIR/factor_factory.log"
        echo "   仪表盘: ./scripts/factor_factory.sh dashboard"
        ;;
    
    stop)
        echo "🛑 发送停止信号..."
        echo "stop" > "$STOP_FILE"
        sleep 2
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "   等待进程退出..."
                sleep 3
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$PID_FILE"
        fi
        rm -f "$STOP_FILE" "$PAUSE_FILE"
        echo "   已停止"
        ;;
    
    pause)
        echo "⏸ 发送暂停信号..."
        echo "pause" > "$PAUSE_FILE"
        echo "   已暂停（使用 resume 恢复）"
        ;;
    
    resume)
        echo "▶ 恢复运行..."
        rm -f "$PAUSE_FILE"
        echo "   已恢复"
        ;;
    
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            PID=$(cat "$PID_FILE")
            UPTIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
            echo "✅ 因子工厂运行中"
            echo "   PID: $PID"
            echo "   运行时间: ${UPTIME:-?}"
            if [ -f "$PAUSE_FILE" ]; then
                echo "   状态: ⏸ 已暂停"
            else
                echo "   状态: ▶ 运行中"
            fi
            echo "   日志: $QLIB_DIR/factor_factory.log"
            echo "   最近进度:"
            if [ -f "$QLIB_DIR/factor_progress.json" ]; then
                "$VENV_PYTHON" -c "
import json
d = json.loads(open('$QLIB_DIR/factor_progress.json').read())
print(f'      阶段: {d.get(\"phase\",\"?\")}  第{d.get(\"round\",0)}轮')
print(f'      强因子: {d.get(\"library_strong\",0)}  候选: {d.get(\"library_candidates\",0)}')
" 2>/dev/null || true
            fi
        else
            echo "💀 因子工厂未运行"
            rm -f "$PID_FILE"
        fi
        ;;
    
    dashboard)
        echo "📊 启动仪表盘（单独窗口）..."
        echo "   提示: 在 Windows 上打开一个新的 CMD 窗口，执行:"
        echo "     wsl ~/TradingAgents-CN/scripts/factor_factory.sh dashboard-live"
        echo ""
        echo "   在本终端中直接启动:"
        "$VENV_PYTHON" "$DASHBOARD"
        ;;
    
    direction)
        if [ -z "${2:-}" ]; then
            if [ -f "$DIRECTION_FILE" ]; then
                echo "🎯 当前挖掘方向:"
                cat "$DIRECTION_FILE"
            else
                echo "🎯 当前挖掘方向: 自动（使用 AI 诊断）"
            fi
            echo ""
            echo "设置新方向:"
            echo "  ./scripts/factor_factory.sh direction \"你的方向描述\""
        else
            echo "$2" > "$DIRECTION_FILE"
            echo "✅ 方向已设置: $2"
        fi
        ;;
    
    *)
        echo "用法: $0 <start|stop|pause|resume|status|dashboard|direction [text]>"
        echo ""
        echo "命令:"
        echo "  start       启动后台因子工厂"
        echo "  stop        停止后台因子工厂"
        echo "  pause       暂停（保留进程）"
        echo "  resume      恢复运行"
        echo "  status      查看状态"
        echo "  dashboard   打开实时仪表盘（本终端）"
        echo "  direction   查看/设置挖掘方向"
        exit 1
        ;;
esac
