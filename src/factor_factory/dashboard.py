#!/usr/bin/env python3
"""
因子工厂实时仪表盘 — 在单独终端窗口运行。

用法:
    python scripts/factor_dashboard.py
    python scripts/factor_dashboard.py --watch  (默认，实时刷新)
    python scripts/factor_dashboard.py --once   (单次快照)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from .paths import PROGRESS_FILE as _PROGRESS_FILE
from .paths import STOP_FILE as _STOP_FILE
from .paths import PAUSE_FILE as _PAUSE_FILE
from .paths import DIRECTION_FILE as _DIRECTION_FILE
from .paths import LIBRARY_FILE as _LIBRARY_FILE

PHASE_EMOJI = {
    "starting": "🚀",
    "running": "🧬",
    "evaluating": "📊",
    "done": "✅",
    "sleeping": "😴",
    "paused": "⏸",
    "stopped": "🛑",
    "error": "❌",
}


def render():
    """Render the dashboard to stdout (ANSI-friendly)."""
    try:
        progress = json.loads(_PROGRESS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {}

    phase = progress.get("phase", "unknown")
    emoji = PHASE_EMOJI.get(phase, "❓")

    # 信号文件状态
    daemon_alive = _is_daemon_alive()

    lines = []
    lines.append("")
    lines.append(f"  {'='*50}")
    lines.append(f"  🧬  因子工厂实时仪表盘")
    lines.append(f"  {'='*50}")
    lines.append("")

    # 状态
    state_str = f"{emoji} {phase.upper()}" if daemon_alive else "💀 守护进程未运行"
    lines.append(f"  状态:   {state_str}")
    lines.append(f"  时间:   {progress.get('updated_at', '-')}")
    lines.append("")

    # 挖掘进度
    strong = progress.get("library_strong", 0)
    candidates = progress.get("library_candidates", 0)
    evaluated = progress.get("factors_evaluated", 0)
    round_num = progress.get("round", 0)
    lines.append(f"  📊 本日第 {round_num} 轮")
    lines.append(f"     已评估: {evaluated} 个因子")
    lines.append(f"     强因子: {strong} 个 ✅")
    lines.append(f"     候选因子: {candidates} 个 ◑")
    lines.append("")

    # 方向
    direction = progress.get("direction", "")
    if not direction:
        try:
            if _DIRECTION_FILE.exists():
                direction = _DIRECTION_FILE.read_text().strip()[:80]
        except Exception:
            pass
    if direction:
        lines.append(f"  🎯 挖掘方向: {direction[:80]}")
    else:
        lines.append(f"  🎯 挖掘方向: 自动（使用上次 AI 诊断）")
    lines.append("")

    # 窗口
    es = progress.get("eval_start", "")
    ee = progress.get("eval_end", "")
    if es:
        lines.append(f"  📅 评估窗口: {es} ~ {ee}")
    lines.append("")

    # 当前正在验证的因子
    cur_expr = progress.get("current_expr", "")
    if cur_expr:
        lines.append(f"  🔬 当前验证: {cur_expr[:70]}")
        lines.append("")

    # 消息/错误
    msg = progress.get("message", "")
    err = progress.get("error", "")
    if msg:
        lines.append(f"  💬 {msg}")
    if err:
        lines.append(f"  ❌ {err}")
        lines.append("")

    # 控制
    lines.append(f"  {'─'*50}")
    lines.append(f"  操作: [S]停止  [P]暂停  [R]恢复  [D]设置方向  [Q]退出")
    lines.append(f"  {'='*50}")
    lines.append("")

    print("\033[H\033[J", end="")  # clear screen
    print("\n".join(lines), end="", flush=True)


def _is_daemon_alive() -> bool:
    """Check if the daemon process is running."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if len(cmdline) >= 2 and "factor_factory_daemon" in " ".join(cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass
    return False


def _handle_key(key: str) -> bool:
    """Handle a key press. Returns False to exit."""
    key = key.upper()
    if key == "S":
        _STOP_FILE.write_text("stop")
        print(f"\n  已发送停止信号")
        return True
    elif key == "P":
        _PAUSE_FILE.write_text("pause")
        print(f"\n  已发送暂停信号")
        return True
    elif key == "R":
        try:
            _PAUSE_FILE.unlink(missing_ok=True)
            print(f"\n  已发送恢复信号")
        except Exception:
            pass
        return True
    elif key == "D":
        print(f"\n  输入挖掘方向（留空取消）:")
        try:
            direction = sys.stdin.readline().strip()
            if direction:
                _DIRECTION_FILE.write_text(direction)
                print(f"  方向已设置: {direction[:80]}")
            else:
                if _DIRECTION_FILE.exists():
                    _DIRECTION_FILE.unlink(missing_ok=True)
                print(f"  方向已被清除（恢复自动诊断）")
        except Exception:
            pass
        time.sleep(1)
        return True
    elif key == "Q":
        return False
    return True


def main_watch():
    """Watch mode — live refresh."""
    import sys
    import tty
    import termios

    print("\033[?25l", end="", flush=True)  # hide cursor
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

        try:
            while True:
                render()
                # Wait for keypress with 2s timeout
                import select
                ready, _, _ = select.select([sys.stdin], [], [], 2)
                if ready:
                    key = sys.stdin.read(1)
                    if key == "\x03":  # Ctrl+C
                        break
                    if not _handle_key(key):
                        break
                # Check if daemon stopped
                if _STOP_FILE.exists() and not _is_daemon_alive():
                    render()
                    time.sleep(1)
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        # Fallback: no keyboard input, just watch
        try:
            while True:
                render()
                time.sleep(2)
                if _STOP_FILE.exists() and not _is_daemon_alive():
                    time.sleep(1)
                    render()
                    break
        except KeyboardInterrupt:
            pass
    finally:
        print("\033[?25h", end="", flush=True)  # show cursor
        print("\n仪表盘已退出")


def main_once():
    """Single snapshot mode."""
    render()
    print("")
    print("  提示: python scripts/factor_dashboard.py --watch 进入实时模式")


if __name__ == "__main__":
    if "--once" in sys.argv:
        main_once()
    else:
        main_watch()
