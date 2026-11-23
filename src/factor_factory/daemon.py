#!/usr/bin/env python3
"""
因子工厂后台守护进程 — 全天候循环挖掘 A 股因子。

控制信号（文件）:
  ~/.qlib/factor_factory.stop   → 停止
  ~/.qlib/factor_factory.pause  → 暂停（检测到此文件则 sleep 重试）
  ~/.qlib/factor_factory.direction.txt → 当前挖掘方向（覆盖诊断）

用法:
    python scripts/factor_factory_daemon.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
_FACTORS_PER_ITER = 15     # 每轮因子数（提案15+变异~5=20，双投票筛到8）
_SLEEP_MINUTES = 0         # 轮间间隔（0=不休眠）
_N_SAMPLE = 200            # IC评估采样股票数
_MAX_DAILY_ROUNDS = 16     # 每天最多16轮
_DAILY_RESET_HOUR = 6      # 每天6点重置计数
_AUDIT_INTERVAL = 30       # 每N轮触发一次PRO审计

# 信号文件路径
from . import paths as _paths
_FF_HOME = _paths.FF_HOME
_STOP_FILE, _PAUSE_FILE = _paths.STOP_FILE, _paths.PAUSE_FILE
_DIRECTION_FILE, _LIBRARY_FILE, _LOG_FILE = _paths.DIRECTION_FILE, _paths.LIBRARY_FILE, _paths.LOG_FILE
_PROGRESS_FILE = _paths.PROGRESS_FILE


_PROGRESS_FILE = _FF_HOME / "factor_progress.json"

# ── 路径 ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 信号工具 ──────────────────────────────────────────────────────
def _should_stop() -> bool:
    return _STOP_FILE.exists()

def _is_paused() -> bool:
    return _PAUSE_FILE.exists()

def _clean_stop_signal():
    if _STOP_FILE.exists():
        _STOP_FILE.unlink(missing_ok=True)

def _read_direction() -> str:
    try:
        if _DIRECTION_FILE.exists():
            return _DIRECTION_FILE.read_text().strip()[:300]
    except Exception:
        pass
    return ""

def _pro_audit(round_num: int) -> str:
    """PRO审计: 分析因子库现状 → 输出下一阶段挖掘方向。

    每 AUDIT_INTERVAL 轮触发一次。
    如果审计成功，写入 direction.txt。
    返回审计产生的方向字符串（空=无变化）。
    """
    try:
        from factor_factory.agent import FactorAgent
        agent = FactorAgent()
        lib = agent.load_library_from_disk()
        if not lib:
            _append_log("  📋 PRO审计: 因子库为空，跳过")
            return ""
        strong = [f for f in lib if f.get("tier") == "strong"]
        candidates = [f for f in lib if f.get("tier") != "strong"]
        # 统计类别分布
        from collections import Counter
        cat_map = {
            "close": "价格动量", "Ref": "价格动量", "Resi": "残差",
            "volume": "量价", "Std": "波动率", "Corr": "相关性",
            "Rank": "排序", "Log": "规模代理",
        }
        cats = Counter()
        for f in lib:
            expr = f.get("expr", "")
            matched = False
            for kw, cat in cat_map.items():
                if kw in expr:
                    cats[cat] += 1
                    matched = True
                    break
            if not matched:
                cats["其他"] += 1
        cat_summary = ", ".join(f"{k}{v}" for k, v in cats.most_common(5))
        prompt = f"""你是一名A股量化因子研究员，负责给因子挖掘团队定方向。

当前因子库: {len(strong)}强 + {len(candidates)}候选 = {len(lib)}个因子
类别分布: {cat_summary}

请分析因子库的不足，输出下阶段挖掘方向。要求:
1. 指出当前库缺少哪类因子（如资金流、波动率偏度、横截面动量等）
2. 给出具体方向描述（30字以内），让Flash能直接按此提案
3. 只输出JSON: {{"direction": "...", "reason": "..."}}"""
        content = agent._call_llm(prompt, temperature=0.3, model_hint="pro")
        import json as _json
        new_dir = ""
        reason = ""
        try:
            parsed = _json.loads(content.strip().lstrip("```json").rstrip("```").strip())
            if isinstance(parsed, dict):
                new_dir = parsed.get("direction", "").strip()
                reason = parsed.get("reason", "")
        except Exception:
            # Bare text fallback: take first non-empty line
            for line in content.split("\n"):
                line = line.strip().strip('"').strip("'")
                if line and not line.startswith("{") and not line.startswith("}") and "direction" not in line:
                    new_dir = line[:30]
                    break
        if new_dir:
            _DIRECTION_FILE.write_text(new_dir, encoding="utf-8")
            _append_log(f"  📋 PRO审计 v{round_num}: {new_dir[:60]}")
            if reason:
                _append_log(f"    原因: {reason[:80]}")
        return new_dir
    except Exception as e:
        _append_log(f"  ⚠ PRO审计失败: {e}")
        return ""

def _append_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── 主循环 ────────────────────────────────────────────────────────
def main():
    _append_log("🧬 因子工厂守护进程启动")
    _append_log(f"   配置: 每轮{_FACTORS_PER_ITER}个因子, 间隔{_SLEEP_MINUTES}分钟, "
                f"采样{_N_SAMPLE}只股票")

    # 清理历史停止信号
    _clean_stop_signal()

    # 写初始进度
    _init_progress()

    round_today = 0
    last_reset_day = datetime.now().day

    while not _should_stop():
        now = datetime.now()

        # 每天重置计数
        if now.day != last_reset_day and now.hour >= _DAILY_RESET_HOUR:
            round_today = 0
            last_reset_day = now.day
            _append_log(f"📅 新的一天，重置轮次计数")

        # 暂停检测
        if _is_paused():
            _update_progress(phase="paused", round=round_today)
            _append_log("⏸ 暂停中... 等待恢复信号")
            time.sleep(10)
            continue

        # 读取当前的挖掘方向
        direction = _read_direction()

        # 每 AUDIT_INTERVAL 轮触发一次PRO审计
        if round_today > 0 and round_today % _AUDIT_INTERVAL == 0:
            _append_log(f"  🔍 PRO审计触发（第{round_today}轮）")
            new_dir = _pro_audit(round_today)
            if new_dir:
                direction = new_dir

        round_today += 1
        _append_log(f"── 第 {round_today} 轮（今日累计 {round_today} 轮）──")
        if direction:
            _append_log(f"   定向挖掘: {direction[:80]}...")

        # 跑一轮因子发现
        try:
            _run_one_cycle(direction, round_today)
        except Exception as e:
            _append_log(f"❌ 本轮异常: {e}")
            _update_progress(phase="error", error=str(e), round=round_today)

        # 检查是否需要停止
        if _should_stop():
            break

        # 写入方向到进度
        _update_progress(
            phase="ready",
            round=round_today,
            direction=direction,
        )

        # 休眠（SLEEP_MINUTES=0 时不休眠，直接下一轮）
        if _SLEEP_MINUTES > 0:
            _append_log(f"😴 休息 {_SLEEP_MINUTES} 分钟...")
            _sleep_with_check(_SLEEP_MINUTES * 60, check_interval=10)
        else:
            # 不休眠时短暂停顿让其他进程有机会执行
            if _should_stop():
                break
            time.sleep(1)

    _clean_stop_signal()
    _update_progress(phase="stopped", round=round_today)
    _append_log("🛑 因子工厂已停止")


def _run_one_cycle(direction: str, round_num: int):
    """跑一轮因子发现。"""
    # 延迟导入（避免启动时加载 qlib + torch）
    from factor_factory.agent import FactorAgent
    import datetime as _dt

    _today = _dt.date.today()
    eval_end = (_today.replace(day=1) - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
    eval_start = "2011-07-05"  # 2010年融资融券推出后，市场结构相对稳定至今

    _update_progress(
        phase="running",
        round=round_num,
        direction=direction,
        eval_start=eval_start,
        eval_end=eval_end,
    )

    agent = FactorAgent()
    result = agent.run_research_loop(
        n_iter=1,
        factors_per_iter=_FACTORS_PER_ITER,
        eval_start=eval_start,
        eval_end=eval_end,
        n_sample=_N_SAMPLE,
        direction=direction,
        enable_mutation=True,
        enable_double_vote=True,
    )

    strong = result.get("factors_strong", 0)
    candidates = result.get("factors_candidate", 0)
    evaluated = result.get("factors_evaluated", 0)

    _append_log(f"  完成: 评估{evaluated}个, 强因子{strong}个, 候选{candidates}个")

    # 更新进度
    _update_progress(
        phase="done",
        round=round_num,
        direction=direction,
        factors_evaluated=evaluated,
        factors_strong=strong,
        factors_candidates=candidates,
    )


def _sleep_with_check(total_seconds: int, check_interval: int = 10):
    """可中断的睡眠：每 check_interval 秒检查一次停止/暂停信号。"""
    elapsed = 0
    while elapsed < total_seconds:
        if _should_stop():
            return
        if _is_paused():
            _append_log("⏸ 检测到暂停，中断休眠")
            _update_progress(phase="paused", message="暂停中")
            while _is_paused() and not _should_stop():
                time.sleep(5)
            if not _should_stop():
                _append_log("▶ 恢复运行")
                elapsed += 5  # 暂停时间不计入睡眠
                continue
            return
        time.sleep(check_interval)
        elapsed += check_interval


def _init_progress():
    """写初始进度文件。"""
    _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _update_progress(phase="starting", message="因子工厂启动中")


def _update_progress(**kwargs):
    """更新进度 JSON 文件（供 dashboard 读取）。"""
    try:
        data = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "library_strong": 0,
            "library_candidates": 0,
            "direction": "",
            "eval_start": "",
            "eval_end": "",
            "factors_evaluated": 0,
            "message": "",
            "error": "",
        }
        data.update(kwargs)

        # 尝试从已有进度文件读取累计因子数（如果 kwargs 没覆盖）
        try:
            if _PROGRESS_FILE.exists():
                old = json.loads(_PROGRESS_FILE.read_text())
                if "library_strong" not in kwargs:
                    data["library_strong"] = old.get("library_strong", 0)
                if "library_candidates" not in kwargs:
                    data["library_candidates"] = old.get("library_candidates", 0)
        except Exception:
            pass

        _PROGRESS_FILE.write_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
