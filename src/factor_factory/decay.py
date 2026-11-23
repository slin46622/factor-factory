#!/usr/bin/env python3
"""
P3: 因子衰减监控 — 定期评估每个因子的IC趋势。

用法:
    python scripts/monitor_factor_decay.py                  # 全面评估
    python scripts/monitor_factor_decay.py --recent-only     # 只看最近100天
    python scripts/monitor_factor_decay.py --force-refresh   # 强制重新计算所有IC

输出:
    ~/.qlib/factor_decay_report.json — 每个因子的近/中期IC对比
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .paths import LIBRARY_FILE as _LIBRARY_FILE
from .paths import DECAY_REPORT_FILE as _REPORT_FILE


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def eval_factor_ic_series(expr: str, n_sample: int = 200) -> dict:
    """对单个因子做IC评估，返回分段IC。

    返回近/中/远三个时间段的IC均值。
    """
    import random
    from pathlib import Path as _P

    data_dir = _P.home() / ".qlib" / "cn_data"
    from qlib.config import REG_CN
    import qlib
    try:
        qlib.init(provider_uri=str(data_dir), region=REG_CN)
    except Exception:
        pass  # already initialized

    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158
    import pandas as pd
    import numpy as np

    # 采样股票
    codes = list(D.list_instruments(instruments="csi500", start_time="2024-01-01", end_time="2026-06-30"))
    if len(codes) > n_sample:
        codes = random.sample(codes, n_sample)
    if not codes:
        return {"error": "no instruments"}

    # 三个时间段
    periods = {
        "近期(3个月)": ("2026-04-01", "2026-06-30"),
        "中期(6个月)": ("2025-10-01", "2026-03-31"),
        "远期(1年+)": ("2024-07-01", "2025-09-30"),
    }

    results = {}
    for pname, (start, end) in periods.items():
        try:
            df = D.features(D.codes(codes), fields=[expr], start_time=start, end_time=end)
            if df is None or df.empty or len(df.columns) < 2:
                results[pname] = {"ic": 0, "n": 0}
                continue
            series = df.iloc[:, 0].values
            # Forward returns (1-day)
            ret = D.features(D.codes(codes), fields=["Ref($close,1)/$close-1"],
                             start_time=start, end_time=end)
            if ret is None or ret.empty:
                results[pname] = {"ic": 0, "n": 0}
                continue
            ret_series = ret.iloc[:, 0].values
            # IC computation
            n = min(len(series), len(ret_series))
            if n < 20:
                results[pname] = {"ic": 0, "n": 0}
                continue
            corr = np.corrcoef(series[:n], ret_series[:n])[0, 1]
            results[pname] = {"ic": round(float(corr), 4), "n": n}
        except Exception:
            results[pname] = {"ic": 0, "n": 0}
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="因子衰减监控")
    parser.add_argument("--force-refresh", action="store_true", help="强制重新计算所有因子IC")
    parser.add_argument("--recent-only", action="store_true", help="只看最近100天")
    args = parser.parse_args()

    if not _LIBRARY_FILE.exists():
        print(f"❌ 因子库文件不存在: {_LIBRARY_FILE}")
        sys.exit(1)

    with open(_LIBRARY_FILE) as f:
        lib = json.load(f)

    print("=" * 60)
    print(f"因子衰减监控 — {len(lib)} 个因子")
    print("=" * 60)

    report = {
        "generated_at": datetime.now().isoformat(),
        "factors": [],
        "summary": {},
    }

    decaying = []
    healthy = []
    unknown = []

    for i, entry in enumerate(lib):
        expr = entry["expr"]
        ic = entry.get("ic_mean", 0)
        tier = entry.get("tier", "?")
        src = entry.get("source", "?")

        # 只评估自定义因子
        if src != "discovered":
            continue

        print(f"\n  [{i+1}/{len(lib)}] IC={ic:.4f} [{tier}] {expr[:60]}")
        sys.stdout.flush()

        ic_series = eval_factor_ic_series(expr)
        if "error" in ic_series:
            print(f"    评估失败: {ic_series['error']}")
            unknown.append(expr)
            continue

        near = ic_series.get("近期(3个月)", {}).get("ic", 0)
        mid = ic_series.get("中期(6个月)", {}).get("ic", 0)
        far = ic_series.get("远期(1年+)", {}).get("ic", 0)

        print(f"    近3月IC: {near:+.4f}  |  6月前IC: {mid:+.4f}  |  1年前IC: {far:+.4f}")

        # 衰减判定
        if abs(near) < 0.01 and abs(mid) > 0.02:
            status = "🔥 严重衰减"
            decaying.append(expr)
        elif abs(near) < 0.5 * abs(mid) and abs(mid) > 0.02:
            status = "⚠ 趋势衰减"
            decaying.append(expr)
        elif abs(near) >= 0.015:
            status = "✅ 健康"
            healthy.append(expr)
        else:
            status = "?- 数据不足"
            unknown.append(expr)

        print(f"    → {status}")

        report["factors"].append({
            "expr": expr[:80],
            "ic_original": ic,
            "tier": tier,
            "ic_near": near,
            "ic_mid": mid,
            "ic_far": far,
            "status": status,
        })

        if not args.recent_only:
            time.sleep(0.5)  # 避免qlib并发问题

    report["summary"] = {
        "total_evaluated": len(report["factors"]),
        "healthy": len(healthy),
        "decaying": len(decaying),
        "unknown": len(unknown),
    }

    _REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print(f"报告已保存: {_REPORT_FILE}")
    print(f"  健康: {len(healthy)}  |  衰减: {len(decaying)}  |  未知: {len(unknown)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
