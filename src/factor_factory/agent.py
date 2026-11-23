"""AI-driven factor discovery and diagnostic reasoning.

Inspired by:
  - R&D-Agent-Quant (NeurIPS 2025): two-phase research→development loop,
    factor-model co-optimization, multi-armed bandit direction selection
  - Gome / 推理即梯度: diagnostic-driven optimization (gradient) vs random
    search (tree search) — the stronger the LLM, the more it benefits from
    structured diagnostics over brute-force enumeration

Core loop
─────────
1. DeepSeek proposes N Qlib factor expressions (hypothesis)
2. We evaluate IC (Information Coefficient) for each on real A-share data
3. Factors with IC > threshold are saved to the library
4. Evaluation results feed the NEXT iteration prompt (Gome momentum)
5. Repeat for n_iter rounds → ranked factor library

IC calculation
──────────────
  IC = daily cross-sectional Pearson corr(factor_value, next-day return)
  IC > 0.02 → predictive;  IC > 0.05 → strong
  ICIR = IC_mean / IC_std (stability measure, > 0.5 is good)
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger("factor_factory.agent")

from .paths import DATA_DIR, LIBRARY_FILE
from . import paths as _paths
_FF_HOME = _paths.FF_HOME
_DEFAULT_DATA_DIR = DATA_DIR

# Factor quality thresholds  (Harvey 2016 / GKX 2020 / McLean & Pontiff 2016 / Feng et al. 2020)
_IC_GOOD = 0.02      # primary tier: IC ≥ 0.02 → "strong" factor
_IC_WEAK = 0.015     # secondary tier: 0.015 ≤ IC < 0.02 → "candidate" factor (High-Flyer ensemble philosophy)
_IC_STRONG = 0.04    # IC ≥ 0.04 → highlight
_ICIR_MIN = 0.3      # Harvey t-stat > 3.0 for 400 days; ICIR > 0.15 minimum, use 0.3
_ICIR_WEAK = 0.2     # looser ICIR for candidate tier
_IC_OOS_RATIO = 0.7  # OOS IC ≥ 70% of IS IC (McLean & Pontiff)
_IC_OOS_WEAK = 0.6   # looser OOS ratio for candidate tier
_CORR_DEDUP = 0.7    # |corr| > 0.7 with higher-IC factor → remove (Feng et al. 2020)
_MULTI_PERIOD_PASS = 2  # GKX: factor must show positive IC at ≥ 2 out of 3 horizons (1d/5d/20d)

# L1 组合回测门槛（多空 Top10%/Bottom10%，年化夏普 + 最大回撤）
_BT_SHARPE_MIN = 0.8    # 强因子：夏普 ≥ 0.8（daily rebalanced long-short, 相当于年化~12%）
_BT_SHARPE_WEAK = 0.5   # 候选因子：夏普 ≥ 0.5
_BT_MDD_MIN = -0.20     # 强因子：最大回撤 ≥ -20%

# Qlib operator/field reference for the LLM prompt
_QLIB_SYNTAX = """\
可用字段：$close $open $high $low $volume $factor

可用算子：
  Ref(field, n)          — 滞后 n 期
  Mean(field, n)         — n 期均值
  Std(field, n)          — n 期标准差
  Max(field, n)          — n 期最大值
  Min(field, n)          — n 期最小值
  Sum(field, n)          — n 期求和
  Rank(field)            — 截面排名（0-1）
  Log(field)             — 自然对数
  Abs(field)             — 绝对值
  Sign(field)            — 符号（-1/0/1）
  Delta(field, n)        — field - Ref(field, n)
  WMA(field, n)          — 加权移动均值
  EMA(field, n)          — 指数移动均值
  Corr(f1, f2, n)        — 滚动 n 期相关系数
  四则运算：+ - * /（避免除零：加 1e-12）

⚠️ 常见错误（必须避免，否则因子无效）：
  错误: -Std(x, n)        → 正确: (-1) * Std(x, n)           # 负号不能直接套函数
  错误: -Log(x+1e-12)     → 正确: (-1) * Log(x+1e-12)        # 负号不能直接套函数
  错误: -Mean(x, n)       → 正确: (-1) * Mean(x, n)           # 负号不能直接套函数
  错误: -Max(x, n)        → 正确: (-1) * Max(x, n)            # 负号不能直接套函数

示例因子（含长周期窗口，参考 GKX 2020 建议 12-36 个月）：
  短期动量：  "Ref($close,5)/$close-1"
  中期动量：  "Ref($close,60)/$close-1"                        # 3个月动量
  长期动量：  "Ref($close,250)/Ref($close,20)-1"               # 12月动量（跳过1月反转）
  反转信号：  "(-1)*($close/Ref($close,5)-1)"                  # 5日反转
  布林位置：  "($close-Mean($close,20))/Std($close,20)"        # 布林带z-score
  成交量比：  "Log(Mean($volume,5)/Mean($volume,60)+1e-12)"    # 短/中期量比
  波动率：    "Std($close/Ref($close,1)-1,60)"                 # 60日实现波动率
  量价散度：  "Corr($close/Ref($close,1)-1,$volume/Mean($volume,20),20)"
  长期趋势：  "Mean($close,20)/Mean($close,250)-1"             # 20日均线偏离长期
  非线性正确写法：  "Rank(($close/Ref($close,60)-1),1) * Rank(((-1)*Std($close/Ref($close,1)-1,20)),1)"

全球鲁棒因子类别（Chen et al. 2023 "Taming the Global Factor Zoo"，48120只股票36个国家验证）：
  1.市场动态类：Beta、已实现波动率、波动率偏度
  2.盈利增长类：盈利加速度代理（近期涨幅加速 vs 历史均值）
  3.质量类：量价质量（价格稳定性 + 成交量稳定性的复合）
  4.动量类：12-1月动量（250日动量跳过最近20日，避免反转污染）
  5.投资类：资产扩张代理（成交量持续放大代理扩张信号）
  6.规模类：成交量×价格的对数（市值替代，越小越好→加负号）
  7.债务/流动性类：换手率异常（异常低换手率代理流动性风险溢价）
"""


class FactorAgent:
    """AI factor discovery and diagnostic reasoning agent."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or str(_DEFAULT_DATA_DIR)).expanduser()
        self._factor_library: list[dict] = []   # accumulated good factors
        self._initialized = False
        self._qlib_codes: list[str] = []        # exchange-prefixed codes
        self._progress: dict = {}               # live progress for status polling

    def get_progress(self) -> dict:
        """Return a snapshot of the current discovery progress."""
        return dict(self._progress)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_research_loop(
        self,
        n_iter: int = 3,
        factors_per_iter: int = 5,
        eval_start: Optional[str] = None,
        eval_end: Optional[str] = None,
        n_sample: int = 200,
        direction: str = "",
        enable_mutation: bool = True,
        enable_double_vote: bool = True,
        cancel_event=None,
    ) -> dict:
        """Run the full R&D-Agent-Quant style research loop.

        Each iteration:
          1. LLM proposes factors (informed by previous results)
          2. We compute IC for each proposed factor (skip if qlib unavailable)
          3. Good factors are saved; results feed the next prompt

        Returns discovered factor library sorted by IC.
        """
        qlib_ok = self._init_qlib()
        if not qlib_ok:
            logger.warning("[FactorAgent] Qlib 未初始化，将运行\"仅提案\"模式（跳过 IC 验证）")

        # 滚动评估窗口：
        #   eval_end   = 上个月末（留 ~1 个月避免 look-ahead）
        #   eval_start = 4 年前（跨越牛/熊多个周期，增强因子鲁棒性）
        #   窗口内 OOS split=70%：约最后 1.2 年作为样本外验证
        import datetime as _dt
        _today = _dt.date.today()
        if eval_end is None:
            eval_end = (_today.replace(day=1) - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
        if eval_start is None:
            eval_start = "2011-07-05"  # 2010年两融后市场结构稳定至今
        logger.info(f"[FactorAgent] IC 评估窗口: {eval_start} ~ {eval_end}")

        # 内存保护：可用内存不足5GB时跳过因子发现（避免 OOM 导致整个后端崩溃）
        try:
            import psutil as _psutil
            _avail_gb = _psutil.virtual_memory().available / 1024**3
            if _avail_gb < 5.0:
                logger.warning(f"[FactorAgent] 可用内存 {_avail_gb:.1f}GB < 5GB，跳过因子发现（防OOM）")
                return {"error": f"可用内存不足（{_avail_gb:.1f}GB），跳过因子发现"}
            logger.info(f"[FactorAgent] 可用内存 {_avail_gb:.1f}GB，开始因子发现")
        except ImportError:
            pass
        self._factor_library = []
        self._factor_candidates: list[dict] = []  # weak tier (High-Flyer ensemble philosophy)
        # Load persisted factors so LLM knows what's already been found
        persisted = self.load_library_from_disk()
        _known_exprs: set[str] = {f["expr"] for f in persisted}
        iteration_history: list[dict] = []
        # Build initial summary from persisted library so the LLM proposes truly new factors
        _persisted_summary = ""
        if persisted:
            top_exprs = [f["expr"][:60] for f in persisted[:5]]
            _persisted_summary = (
                f"【注意】已有 {len(persisted)} 个因子保存在磁盘上，请勿重复提案以下已知因子：\n"
                + "\n".join(
                    f"  - {f['expr'][:55]:<55s}  IC={f.get('ic_mean',0):+.4f} ICIR={f.get('icir',0):+.2f} ({f.get('tier','?')})"
                    for f in persisted[:5]
                )
                + "\n请提案与上述因子低相关、高互补的全新表达式。\n"
            )
        total_factors = n_iter * factors_per_iter
        self._progress = {
            "current_iter": 0,
            "total_iters": n_iter,
            "evaluated": 0,
            "total_factors": total_factors,
            "found": 0,
            "current_expr": "",
            "phase": "starting",
        }

        logger.info(f"[FactorAgent] 开始 R&D 因子发现循环: {n_iter} 轮 × {factors_per_iter} 因子")

        for it in range(1, n_iter + 1):
            if cancel_event is not None and cancel_event.is_set():
                logger.info(f"[FactorAgent] 收到取消信号，在第 {it} 轮停止")
                break
            logger.info(f"[FactorAgent] ── 第 {it} 轮 ──")
            self._progress["current_iter"] = it
            self._progress["phase"] = "proposing"
            self._progress["current_expr"] = ""

            # Build prompt with feedback from previous iterations (Gome momentum)
            prev_summary = _persisted_summary + self._summarize_history(iteration_history)
            # Inject latest AI diagnosis direction (compact, ~200 chars, never accumulates)
            diagnosis_dir = self.load_diagnosis_compact()
            # 如果外部传了 direction，优先使用（覆盖默认诊断）
            if direction:
                diagnosis_dir = direction
            if diagnosis_dir and it == 1:
                prev_summary = diagnosis_dir + "\n" + prev_summary
            exprs = self.propose_factors(
                n=factors_per_iter,
                iteration=it,
                prev_summary=prev_summary,
                diagnosis_direction=diagnosis_dir if it <= 2 else "",
            )
            if not exprs:
                logger.warning(f"[FactorAgent] 第 {it} 轮未收到因子提案，跳过")
                continue

            # Skip already-known expressions (from disk)
            new_exprs = [e for e in exprs if e not in _known_exprs]
            skipped = len(exprs) - len(new_exprs)
            if skipped:
                logger.info(f"[FactorAgent] 第 {it} 轮跳过 {skipped} 个已知因子")
            _known_exprs.update(new_exprs)

            # ── C 因子变异：从已有强因子生成变异表达式 ──────────
            if enable_mutation:
                mutations = self._generate_mutations(existing=new_exprs, n=5)
                for m in mutations:
                    if m not in _known_exprs:
                        new_exprs.append(m)
                        _known_exprs.add(m)
                if mutations:
                    logger.info(f"[FactorAgent]  + 因子变异: {len(mutations)} 个")

            # ── D 双LLM投票：第二个Flash筛选Top-N ──────────────
            if enable_double_vote and len(new_exprs) > 8:
                self._progress["phase"] = "voting"
                vote_n = 8
                ranked = self._double_vote_rank(new_exprs, top_n=vote_n, direction=direction)
                if ranked and len(ranked) < len(new_exprs):
                    logger.info(f"[FactorAgent]  + 双投票: {len(new_exprs)}→{len(ranked)} 个")
                    new_exprs = ranked

            self._progress["phase"] = "evaluating"

            # ── A 表达式清洗：修复已知语法错误 ─────────────────
            sanitized = []
            syntax_errors_fixed = 0
            for expr in new_exprs:
                clean = self._sanitize_expression(expr)
                if clean != expr:
                    syntax_errors_fixed += 1
                    logger.debug(f"[FactorAgent]  清洗: {expr[:50]} → {clean[:50]}")
                sanitized.append(clean)
            if syntax_errors_fixed:
                logger.info(f"[FactorAgent]  + 清洗: 修复 {syntax_errors_fixed}/{len(new_exprs)} 个表达式语法")
            self._progress["syntax_errors_fixed"] = syntax_errors_fixed
            new_exprs = sanitized

            # Evaluate each proposed factor
            iter_results = []
            for i, expr in enumerate(new_exprs):
                self._progress["current_expr"] = expr[:80]
                self._progress["evaluated"] = (it - 1) * factors_per_iter + i

                if qlib_ok:
                    metrics = self._evaluate_factor_ic(expr, eval_start, eval_end, n_sample)
                else:
                    metrics = {
                        "ic_mean": 0, "icir": 0, "ic_std": 0, "n_days": 0,
                        "oos_ic_mean": 0, "ic_5d": 0, "ic_20d": 0,
                        "multi_period_pass": 0, "error": "qlib 未安装，跳过 IC 验证"
                    }
                entry = {
                    "expr": expr,
                    "iteration": it,
                    **metrics,
                }
                iter_results.append(entry)
                ic   = metrics.get("ic_mean", 0)
                icir = metrics.get("icir", 0)
                oos_ic = metrics.get("oos_ic_mean", 0)
                mp   = metrics.get("multi_period_pass", 0)   # GKX multi-period horizons

                if not qlib_ok:
                    # 提案模式：所有 LLM 提案都保存为"候选"（由用户后续验证）
                    quality = "📝 提案"
                    entry["tier"] = "candidate"
                    self._factor_candidates.append(entry)
                    self._progress["found"] = len(self._factor_library) + len(self._factor_candidates)
                    logger.info(f"[FactorAgent]   {quality}  {expr[:50]}")
                    continue

                # Primary tier (strong): IC ≥ 0.02, ICIR ≥ 0.3, OOS ≥ 70%, ≥2 horizons
                # L1 add: Sharpe ≥ 0.8, max DD ≥ -20%
                bt_sharpe = metrics.get("bt_sharpe", 0.0)
                bt_mdd = metrics.get("bt_max_dd", 0.0)
                passes_primary = (
                    ic >= _IC_GOOD
                    and icir >= _ICIR_MIN
                    and oos_ic >= _IC_GOOD * _IC_OOS_RATIO
                    and mp >= _MULTI_PERIOD_PASS
                    and bt_sharpe >= _BT_SHARPE_MIN
                    and bt_mdd >= _BT_MDD_MIN
                )
                # Secondary (candidate): IC ≥ 0.015, ICIR ≥ 0.2, OOS ≥ 60%
                # L1 add: Sharpe ≥ 0.5
                passes_candidate = (
                    not passes_primary
                    and ic >= _IC_WEAK
                    and icir >= _ICIR_WEAK
                    and oos_ic >= _IC_WEAK * _IC_OOS_WEAK
                    and bt_sharpe >= _BT_SHARPE_WEAK
                )

                if ic >= _IC_STRONG and passes_primary:
                    quality = "✅ 优"
                elif passes_primary:
                    quality = "✓ 强"
                elif passes_candidate:
                    quality = "◑ 候选"
                elif ic >= _IC_GOOD:
                    quality = "⚠ 不稳" if icir < _ICIR_MIN else ("⚠ OOS衰减" if oos_ic < _IC_GOOD * _IC_OOS_RATIO else ("⚠ 夏普低" if bt_sharpe < _BT_SHARPE_WEAK else "⚠ 单周期"))
                else:
                    quality = "✗ 弱"

                logger.info(
                    f"[FactorAgent]   {quality} IC={ic:.4f} ICIR={icir:.4f} "
                    f"OOS={oos_ic:.4f} MP={mp}/3 Sharpe={bt_sharpe:.2f} DD={bt_mdd:.1%}  {expr[:50]}"
                )
                if passes_primary:
                    entry["tier"] = "strong"
                    self._factor_library.append(entry)
                    self._progress["found"] = len(self._factor_library)
                elif passes_candidate:
                    entry["tier"] = "candidate"
                    self._factor_candidates.append(entry)
            self._progress["evaluated"] = it * factors_per_iter

            iteration_history.append({
                "iteration": it,
                "results": iter_results,
                "good_count": sum(1 for r in iter_results if r.get("ic_mean", 0) >= _IC_GOOD),
            })
            self.save_progress()

        # Deduplicate correlated factors (Feng et al.: keep higher-IC when |corr| > threshold)
        self._deduplicate_library()

        # Sort both tiers by IC
        self._factor_library.sort(key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)
        self._factor_candidates.sort(key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)

        self._progress["phase"] = "done"
        self._progress["current_expr"] = ""
        logger.info(
            f"[FactorAgent] 完成: 强因子 {len(self._factor_library)} 个 + "
            f"候选因子 {len(self._factor_candidates)} 个（去重后）"
        )

        # Strip ic_series (only needed for dedup) before persisting
        for f in self._factor_library + self._factor_candidates:
            f.pop("ic_series", None)

        # Persist to disk so training can use them
        try:
            self.save_library()
        except Exception as exc:
            logger.warning(f"[FactorAgent] 因子库保存失败（不影响结果）: {exc}")
        self.save_progress()
        self.save_feedback(iteration_history)

        return {
            "ok": True,
            "iterations": n_iter,
            "factors_evaluated": sum(len(h["results"]) for h in iteration_history),
            "factors_strong": len(self._factor_library),
            "factors_candidate": len(self._factor_candidates),
            "factors_discovered": len(self._factor_library) + len(self._factor_candidates),
            "library": self._factor_library,
            "candidates": self._factor_candidates,
        }

    _window_funcs = ['Min', 'Max', 'Mean', 'Std', 'Sum', 'Corr', 'Cov', 'WMA', 'EMA', 'Kurt', 'Skew']

    @staticmethod
    def _sanitize_expression(expr: str) -> str:
        """修复Flash生成的Qlib表达式中的常见语法错误。

        已知错误类型：
        1. Min/Max/Mean/Std/Sum(..., 0) → (... , 5)     — 窗口0无效
        2. Rank(expr) → Rank(expr, 1)                    — 缺少N参数
        3. -($expr) → (-1)*($expr)                       — 函数前直接负号
        """
        result = expr.strip()

        # 1. Fix Func(..., 0) when 0 is the last arg and func is a window-aware operator
        import re as _re
        for func in FactorAgent._window_funcs:
            idx = 0
            while True:
                idx = result.find(func + '(', idx)
                if idx == -1:
                    break
                start = idx + len(func) + 1  # position after '('
                depth = 1
                i = start
                while i < len(result) and depth > 0:
                    if result[i] == '(':
                        depth += 1
                    elif result[i] == ')':
                        depth -= 1
                    i += 1
                if depth != 0:
                    idx += 1
                    continue
                end = i - 1  # position of matching ')'
                inner = result[start:end]
                last_comma = inner.rfind(',')
                if last_comma != -1:
                    after = inner[last_comma + 1:].strip()
                    if after == '0':
                        fix_end = start + last_comma + 1 + len(after)
                        result = result[:fix_end - len(after)] + ' 5' + result[fix_end:]
                        # Recalculate because result length changed
                        break
                idx += 1

        # 2. Fix Rank(expr) → Rank(expr, 1)
        idx = 0
        while True:
            idx = result.find('Rank(', idx)
            if idx == -1:
                break
            start = idx + 5
            depth = 1
            i = start
            while i < len(result) and depth > 0:
                if result[i] == '(':
                    depth += 1
                elif result[i] == ')':
                    depth -= 1
                i += 1
            if depth != 0:
                idx += 1
                continue
            end = i - 1
            inner = result[start:end]
            # Check for comma at depth 0
            has_arg_sep = False
            d = 0
            for c in inner:
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif c == ',' and d == 0:
                    has_arg_sep = True
                    break
            if not has_arg_sep:
                # No second arg — insert ',1' before ')'
                result = result[:end] + ', 1' + result[end:]
                idx = end + len(', 1')  # skip past insert
            else:
                idx += 1

        # 3. Fix bare minus prefix: -($expr) → (-1)*($expr)
        result = _re.sub(r'^-\(', '(-1)*(', result)
        # Also in the middle of expressions: ... -($close mean) ...
        result = _re.sub(r'(?<=[+\-*/\(,])-\(', '(-1)*(', result)

        return result

    def _generate_mutations(self, existing: list[str], n: int = 5) -> list[str]:
        """从已有强因子生成变异表达式（窗口调整/算子替换/组合）。

        从 factor_library.json 读取强因子，随机选 n 个最强者做变异。
        新增组合杂交：Rank(f1)*Rank(f2) 等交叉组合。
        """
        import random
        persisted = self.load_library_from_disk(tier="strong")
        if not persisted:
            return []
        # 取 top N 个强因子作为变异源
        sources = sorted(persisted, key=lambda x: -abs(x.get("ic_mean", 0)))[:5]
        mutations: list[str] = []
        _operators = ["Mean", "Std", "Max", "Min", "Sum", "WMA", "EMA"]
        _windows = [5, 10, 20, 40, 60, 120, 250]

        # 先预留组合杂交名额
        hybrid_slots = max(1, n // 4)
        window_slots = n - hybrid_slots

        for src in sources:
            expr = src.get("expr", "")
            if len(mutations) >= window_slots:
                break
            # 变异1: 替换窗口数字
            import re
            tokens = re.findall(r"\d+", expr)
            for t in tokens:
                if len(mutations) >= n:
                    break
                old_val = int(t)
                new_val = random.choice([w for w in _windows if w != old_val])
                mutated = expr.replace(t, str(new_val), 1)
                if mutated != expr and mutated not in existing and mutated not in mutations:
                    mutations.append(mutated)
            if len(mutations) >= n:
                break
            # 变异2: 替换算子
            for op in _operators:
                if op in expr and len(mutations) < n:
                    new_op = random.choice([o for o in _operators if o != op])
                    mutated = expr.replace(op, new_op, 1)
                    if mutated != expr and mutated not in existing and mutated not in mutations:
                        mutations.append(mutated)
            if len(mutations) >= n:
                break

        # 变异3: 组合杂交 — 取两个不同强因子做 Rank 交叉组合
        if len(sources) >= 2 and len(mutations) < n:
            _combos = [
                f"Rank({sources[0]['expr']}, 1) * Rank({sources[1]['expr']}, 1)",
                f"Rank({sources[0]['expr']}, 1) - Rank({sources[1]['expr']}, 1)",
                f"Sign({sources[0]['expr']}) * Abs({sources[1]['expr']})",
            ]
            if len(sources) >= 3:
                _combos.append(
                    f"Rank({sources[0]['expr']}, 1) * Rank({sources[2]['expr']}, 1)"
                )
            for combo in _combos:
                if len(mutations) >= n:
                    break
                if combo not in existing and combo not in mutations:
                    mutations.append(combo)
        if len(mutations) > window_slots:
            logger.info(f"[FactorAgent]  组合杂交: {len(mutations) - window_slots} 个 (Rank交叉)")
        logger.info(f"[FactorAgent]  因子变异: 从{len(sources)}个源生成{len(mutations)}个变异")
        return mutations[:n]

    def _double_vote_rank(self, candidates: list[str], top_n: int = 10, direction: str = "") -> list[str]:
        """第二个LLM实例对候选因子排序，只保留最可能高IC的Top-N。"""
        if len(candidates) <= top_n:
            return candidates
        dir_hint = f"\n当前研究方向: {direction}\n" if direction else ""
        prompt = f"""你是量化因子研究员，请从以下 {len(candidates)} 个 Qlib 因子表达式中，
选出最可能具有高预测力（高IC）的 {top_n} 个。
优先选择：逻辑清晰、窗口适中、未被 Alpha158 覆盖的因子。
{dir_hint}
直接返回 JSON: {{"ranked": ["expr1", "expr2", ...]}}
因子列表:
""" + "\n".join(f"{i+1}. {e}" for i, e in enumerate(candidates))
        try:
            content = self._call_llm(prompt, temperature=0.3, model_hint="flash")
            parsed = self._parse_json_from_llm(content)
            ranked = parsed.get("ranked", [])
            # 与原始列表交集保序
            valid = [e for e in ranked if e in candidates][:top_n]
            if valid:
                return valid
        except Exception as e:
            logger.warning(f"[FactorAgent] 双投票失败: {e}")
        return candidates[:top_n]

    def propose_factors(
        self,
        n: int = 5,
        iteration: int = 1,
        prev_summary: str = "",
        diagnosis_direction: str = "",
    ) -> list[str]:
        """Ask DeepSeek to propose N Qlib factor expressions.

        Returns a list of expression strings; empty list on failure.
        """
        # ── D: 论文分类约束 ────────────────────────────────────────
        _DIRECTION_TAXONOMY = {
            "动量加速度": "本期聚焦「动量加速度类」因子：短期过±20日后多空反转、DGTW残差动量、12-1月动量衰减、因子动量。避免均值回复剥离。",
            "波动率偏度": "本期聚焦「波动率偏度类」因子：日内波动率变化、日间波动率偏度、下行波动率/上行波动率比、最大回撤恢复时间。关注左尾风险和肥尾效应。",
            "资金流异常": "本期聚焦「资金流异常类」因子：资金流(OBV/CMF/MFI)突变、资金流与价格背离、资金集中度变化、大宗交易占比。",
            "流动性突变": "本期聚焦「流动性突变类」因子：Amihud非流动性指标变化、换手率加速/衰减、买卖价差代理、成交量滚动Z-score异常值。",
            "质量代理": "本期聚焦「质量代理类」因子：盈利稳定性的价格代理、ROE代理、毛利率稳定性代理、杠杆变化。无法获取财报时用量价代理。",
            "横截面相关性": "本期聚焦「横截面相关性类」因子：个股对市场/行业/因子的暴露变化，如Beta变化、特质波动率、行业集中度。用多空组合捕捉异象。",
        }
        direction_block = ""
        if diagnosis_direction:
            # Try to match a known taxonomy category
            _matched_taxonomy = ""
            for _cat, _desc in _DIRECTION_TAXONOMY.items():
                if any(kw in diagnosis_direction for kw in [_cat, _cat[:2], _cat[2:4]]):
                    _matched_taxonomy = _desc
                    break
            direction_block = (
                f"\n【诊断反馈】{diagnosis_direction}\n"
                + (_matched_taxonomy + "\n" if _matched_taxonomy else "")
                + "请仅沿此方向提案，不要跨类别发散。\n"
            )

        # ── 构建因子库参考：已存在的因子及其 IC/ICIR（给 LLM 避免重复提案） ──
        _lib_ref = ""
        try:
            persisted = self.load_library_from_disk()
            if persisted:
                lines = [f"当前因子库共 {len(persisted)} 个因子（强因子优先展示）："]
                for f in persisted:
                    expr = f.get("expr", "?")
                    ic = f.get("ic_mean", 0)
                    icir = f.get("icir", 0)
                    oos = f.get("oos_ic_mean", 0)
                    mp = f.get("multi_period_pass", 0)
                    tier = f.get("tier", "?")
                    # 只有 discovered 的因子才有 iteration 字段
                    src = "discovered" if "iteration" in f else "pre-seeded"
                    lines.append(
                        f"  [{tier:>9}][{src}] IC={ic:+.4f} ICIR={icir:+.2f} "
                        f"OOS={oos:+.4f} MP={mp}/3  {expr[:80]}"
                    )
                _lib_ref = "\n".join(lines)
        except Exception:
            _lib_ref = ""

        # ── 模型特征重要性注入（L2a 分析结果，已存在时使用） ─────
        _fi_report = ""
        try:
            _fi_file = _FF_HOME / "feature_importance_report.json"
            if _fi_file.exists():
                _fi_data = json.loads(_fi_file.read_text())
                _top = _fi_data.get("top_features_alpha158", [])
                if _top:
                    _lines = [
                        "【当前 Ensemble 模型特征重要性 Top 15】（来自 18 个 Alpha158 模型的归因分析）",
                        "这些特征已在模型中证明有效，请提案与之不同角度、低相关的新方向："
                    ]
                    for f in _top[:15]:
                        _lines.append(
                            f"  #{f['rank']:>2d} {f['name']:<12s} 重要性{f['importance_pct']:>5.1f}% "
                            f"覆盖{f['model_coverage']}  {f.get('description','')}"
                        )
                    _fi_report = "\n".join(_lines)
        except Exception:
            pass
        _fail_analysis = ""
        # ── B: 自反馈（跨轮次持久化，提醒Flash此前表现）──
        _fb = self.load_feedback()
        if _fb:
            _fail_analysis = _fb

        # ── A: 学术论文模板 ───────────────────────────────────────
        _paper_block = ""
        try:
            _template_file = Path(__file__).parent / "paper_templates.json"
            if _template_file.exists():
                _templates = json.loads(_template_file.read_text())
                _lines = ["\n【学术论文因子模板库】请从以下模板出发做转写适配，不要从零发明："]
                for _cat, _data in _templates.items():
                    _label = _data.get("label", _cat)
                    _papers = ", ".join(_data.get("papers", []))
                    _lines.append(f"\n  -- {_label} ({_papers}) --")
                    for _t in _data.get("templates", [])[:4]:
                        _lines.append(f"     - {_t['name']}: {_t['expr'][:55]}")
                _paper_block = "\n".join(_lines)
        except Exception:
            pass
        if prev_summary:
            # 从迭代历史中统计失败情况
            _rejected_examples = []
            _rejected_ic_low = 0
            _rejected_icir_low = 0
            _rejected_mp_low = 0
            _total_rejected = 0
            for line in prev_summary.split("\n"):
                if line.startswith("  IC=") and "ICIR=" in line:
                    _total_rejected += 1
                    parts = line.split()
                    try:
                        ic_val = float(line.split("IC=")[1].split()[0])
                        icir_val = float(line.split("ICIR=")[1].split()[0])
                    except (ValueError, IndexError):
                        continue
                    if ic_val < _IC_GOOD:
                        _rejected_ic_low += 1
                    if icir_val < _ICIR_MIN:
                        _rejected_icir_low += 1
                    _rejected_examples.append(line[:80])
            if _total_rejected > 0:
                pct_ic = _rejected_ic_low / _total_rejected * 100
                pct_icir = _rejected_icir_low / _total_rejected * 100
                _fail_analysis = (
                    f"\n【上一轮淘汰分析】（共 {_total_rejected} 个淘汰因子）\n"
                    f"  - IC<{_IC_GOOD}（预测力不足）: {_rejected_ic_low} 个 ({pct_ic:.0f}%)\n"
                    f"  - ICIR<{_ICIR_MIN}（不稳定）: {_rejected_icir_low} 个 ({pct_icir:.0f}%)\n"
                    f"  - 多周期未通过: 其余淘汰\n"
                    f"请避免上述失效模式，优先探索长周期、非线性交互方向。\n"
                )

        prompt = f"""你是一位资深 A 股量化因子研究员，熟悉 Gu-Kelly-Xiu 2020、Harvey et al. 2016、McLean-Pontiff 2016 等核心量化论文。
请提出 {n} 个用于预测下一日股票收益的 Qlib 因子表达式。

{_QLIB_SYNTAX}

{_lib_ref}

{_fi_report}

{_fail_analysis}

{_paper_block}

{"迭代上下文（前几轮结果，请在此基础上改进或探索新方向）：" + prev_summary if prev_summary else "这是第一轮探索，请覆盖多种不同的因子类型。"}
{direction_block}
研究要求（GKX 2020 + Harvey 2016 + Global Factor Zoo 2023）：
1. 每个因子必须是合法的 Qlib 表达式字符串
2. **必须提案与因子库中现有因子低相关（不同逻辑/不同窗口/不同算子组合）的全新因子**，请勿重复
3. 优先基于模板库的论文已知因子方向做转写适配（改窗口/加Rank/组合修改），不要从零发明
4. 鼓励 **Rank 交叉组合**：Rank(A)*Rank(B) 等横截面排名交互
5. 每轮至少覆盖以下全球鲁棒类别之一（未被 Alpha158 覆盖的新角度）：
   - 长期动量：12月-1月动量 Ref($close,250)/Ref($close,20)-1
   - 质量代理：价格稳定性 (-1)*Std($close/Ref($close,1)-1,120)（低波动=高质量）
   - 规模代理：(-1)*Log(Mean($volume,60)*Mean($close,60)+1e-12)（小市值溢价）
   - 流动性异常：Mean($volume,5)/Mean($volume,250)-1（换手率突变）
   - 盈利动量：成交量加速 Mean($volume,20)/Mean($volume,120)-1（资金关注度）
6. 因子值越大应越倾向于上涨；若方向相反请加负号（用(-1)*expr，不可在函数前直接加负号）
7. 如果前几轮提案大部分因 IC 不够被拒，请尝试全新方向，不要在同一个小类别里反复试
8. 返回纯 JSON，不要其他任何内容：
{{"factors": ["expr1", "expr2", ...]}}
"""
        try:
            content = self._call_llm(prompt, temperature=0.85, model_hint="flash")
            parsed = self._parse_json_from_llm(content)
            exprs = parsed.get("factors", [])
            if isinstance(exprs, list):
                # Basic sanity: filter empty strings
                return [str(e).strip() for e in exprs if e and str(e).strip()]
        except Exception as exc:
            logger.warning(f"[FactorAgent] 因子提案解析失败: {exc}，重试（简化prompt）")
            # 重试：简化prompt，去掉反馈/库引用
            simple_prompt = f"""提出 {n} 个Qlib因子表达式预测下一日收益。
格式: {{"factors": ["expr1", "expr2", ...]}}
要求: 表达式合法、与Alpha158低相关、窗口72-750日。
{direction_block}"""
            try:
                content = self._call_llm(simple_prompt, temperature=0.7, model_hint="flash")
                parsed = self._parse_json_from_llm(content)
                exprs = parsed.get("factors", [])
                if isinstance(exprs, list):
                    return [str(e).strip() for e in exprs if e and str(e).strip()]
            except Exception as exc2:
                logger.warning(f"[FactorAgent] 重试仍失败: {exc2}")
        return []

    def diagnose(
        self,
        backtest_result: Optional[dict] = None,
        selection_result: Optional[dict] = None,
    ) -> dict:
        """Gome-style diagnostic reasoning.

        Instead of a scalar score, feeds the full result to DeepSeek to
        extract structured improvement directions — 'gradient' rather than
        random exploration.

        Returns structured diagnosis with root_cause, strengths,
        weaknesses, and 3 concrete recommendations.
        """
        ctx_parts = []
        if backtest_result:
            ctx_parts.append(f"回测结果：\n{json.dumps(backtest_result, ensure_ascii=False, indent=2)}")
        if selection_result:
            top5 = (selection_result.get("stocks") or [])[:5]
            ctx_parts.append(
                f"最新选股 Top-5（共 {selection_result.get('total', 0)} 只）：\n"
                + json.dumps(top5, ensure_ascii=False, indent=2)
            )
        if self._factor_library:
            top3 = self._factor_library[:3]
            ctx_parts.append(
                f"已发现的高质量因子 Top-3：\n"
                + json.dumps([{"expr": f["expr"], "ic": f.get("ic_mean")} for f in top3],
                             ensure_ascii=False)
            )

        if not ctx_parts:
            return {"error": "没有结果可以诊断，请先运行选股或回测"}

        today = datetime.today().strftime("%Y-%m-%d")
        prompt = f"""你是一个量化策略诊断专家（参考 Gome 论文的诊断驱动优化方法）。
分析以下 A 股量化策略执行结果，给出结构化诊断，重点是"为什么"，而不只是复述数字。

日期：{today}
{chr(10).join(ctx_parts)}

请按如下 JSON 格式返回诊断报告（纯 JSON，不要额外说明）：
{{
  "overall": "一句话总结：优秀/良好/一般/较差",
  "root_cause": "策略表现的核心原因（2-3句话，深入分析驱动因素）",
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["劣势1", "劣势2"],
  "recommendations": [
    {{"action": "具体操作1", "reason": "原因", "priority": "高/中/低"}},
    {{"action": "具体操作2", "reason": "原因", "priority": "高/中/低"}},
    {{"action": "具体操作3", "reason": "原因", "priority": "高/中/低"}}
  ],
  "next_factor_direction": "下一步因子研究方向建议"
}}
"""
        try:
            content = self._call_llm(prompt, temperature=0.3, model_hint="pro")
            result = self._parse_json_from_llm(content)
            # Save compact diagnosis for Factor Lab feedback (overwrites, no accumulation)
            self.save_diagnosis_compact(result)
            logger.info("[FactorAgent] 诊断完成")
            return {"ok": True, "diagnosis": result}
        except Exception as exc:
            logger.error(f"[FactorAgent] 诊断失败: {exc}")
            return {"error": str(exc)}

    def get_library(self) -> list[dict]:
        """Return the current factor library sorted by IC."""
        return self._factor_library

    # ------------------------------------------------------------------
    # Deduplication (Feng et al. / portfolio-theory: uncorrelated factors add more info)
    # ------------------------------------------------------------------

    def _deduplicate_library(self) -> None:
        """Remove redundant factors whose IC series correlates > _CORR_DEDUP with a better factor.

        Keeps the higher-IC factor in each correlated pair.  Operates in-place on
        self._factor_library.  Factors without ic_series are kept unconditionally.
        """
        import numpy as np

        if len(self._factor_library) < 2:
            return

        # Sort by |IC| descending so we always keep the better one
        self._factor_library.sort(key=lambda x: abs(x.get("ic_mean", 0)), reverse=True)
        kept: list[dict] = []
        dropped = 0
        for candidate in self._factor_library:
            series_c = candidate.get("ic_series", [])
            if not series_c:
                kept.append(candidate)
                continue
            arr_c = np.array(series_c, dtype=float)
            redundant = False
            for existing in kept:
                series_e = existing.get("ic_series", [])
                if not series_e:
                    continue
                arr_e = np.array(series_e, dtype=float)
                # Align to the shorter series
                n = min(len(arr_c), len(arr_e))
                if n < 20:
                    continue
                corr = float(np.corrcoef(arr_c[:n], arr_e[:n])[0, 1])
                if abs(corr) > _CORR_DEDUP:
                    redundant = True
                    dropped += 1
                    logger.info(
                        f"[FactorAgent] 去重: |corr|={abs(corr):.2f} 与 [{existing['expr'][:40]}]，"
                        f"移除 [{candidate['expr'][:40]}] (IC={candidate.get('ic_mean',0):.4f})"
                    )
                    break
            if not redundant:
                kept.append(candidate)

        self._factor_library = kept
        if dropped:
            logger.info(f"[FactorAgent] 相关性去重: 移除 {dropped} 个冗余因子，剩余 {len(kept)} 个")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _LIBRARY_FILE = _FF_HOME / "factor_library.json"
    _DIAGNOSIS_COMPACT_FILE = _FF_HOME / "last_diagnosis_compact.json"
    _PROGRESS_FILE = _FF_HOME / "factor_progress.json"
    _FEEDBACK_FILE = _FF_HOME / "factor_feedback.json"
    _SNAPSHOT_DIR = _FF_HOME / "snapshots"
    _MAX_ROUNDS_FEEDBACK = 10  # only keep recent N rounds in feedback
    _MAX_SNAPSHOTS = 10

    @classmethod
    def _snapshot_before(cls, path: Path) -> None:
        """备份文件到快照目录。只保留最近 _MAX_SNAPSHOTS 个。"""
        if not path.exists():
            return
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = cls._SNAPSHOT_DIR / ts / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(path), str(dest))
            snapshots = sorted(cls._SNAPSHOT_DIR.iterdir()) if cls._SNAPSHOT_DIR.exists() else []
            while len(snapshots) > cls._MAX_SNAPSHOTS:
                oldest = snapshots.pop(0)
                shutil.rmtree(str(oldest), ignore_errors=True)
        except Exception as e:
            logger.warning(f"[FactorAgent] 快照备份失败 ({path.name}): {e}")

    def save_progress(self) -> None:
        """写进度到 ~/.qlib/factor_progress.json（供 dashboard 读取）。"""
        try:
            self._PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            # 始终从磁盘读取真实累计数据
            disk_lib = self.load_library_from_disk()
            disk_strong = sum(1 for f in disk_lib if f.get("tier") == "strong")
            p = dict(self._progress)
            p["library_strong"] = disk_strong
            p["library_candidates"] = len(disk_lib) - disk_strong
            p["factors_evaluated"] = p.get("evaluated", 0)
            self._PROGRESS_FILE.write_text(json.dumps(p, ensure_ascii=False))
        except Exception:
            pass

    def save_feedback(self, iteration_history: Optional[list] = None) -> None:
        """写反馈历史，包含本轮失败原因分类。

        分类依据:
        - syntax_error: 语法错误（被清洗修复的）
        - ic_too_low: IC < 0.02
        - icir_too_low: ICIR < 0.3
        - oos_decay: OOS IC < IS IC × 0.7
        - mp_failed: 多周期通过数 < 2
        """
        try:
            p = dict(self._progress)
            # ── 分析失败原因 ──────────────────────────────
            fail_reasons: dict[str, int] = {
                "syntax_error": p.get("syntax_errors_fixed", 0),
                "ic_too_low": 0,
                "icir_too_low": 0,
                "oos_decay": 0,
                "mp_failed": 0,
            }
            if iteration_history:
                for h in iteration_history:
                    for r in h.get("results", []):
                        ic = r.get("ic_mean", 0)
                        icir = r.get("icir", 0)
                        oos = r.get("oos_ic_mean", 0)
                        mp = r.get("multi_period_pass", 0)
                        if ic < _IC_GOOD:
                            fail_reasons["ic_too_low"] += 1
                        elif icir < _ICIR_MIN:
                            fail_reasons["icir_too_low"] += 1
                        elif oos < _IC_GOOD * _IC_OOS_RATIO:
                            fail_reasons["oos_decay"] += 1
                        elif mp < _MULTI_PERIOD_PASS:
                            fail_reasons["mp_failed"] += 1
            record = {
                "round": p.get("round", 0),
                "errors_fixed": p.get("syntax_errors_fixed", 0),
                "fail_reasons": fail_reasons,
                "strong_count": len(self._factor_library),
                "candidate_count": len(self._factor_candidates),
                "evaluated": p.get("evaluated", 0),
            }
            self._FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            history = []
            if self._FEEDBACK_FILE.exists():
                history = json.loads(self._FEEDBACK_FILE.read_text())
            history.append(record)
            history = history[-self._MAX_ROUNDS_FEEDBACK:]
            self._FEEDBACK_FILE.write_text(json.dumps(history, ensure_ascii=False))
        except Exception:
            pass

    @classmethod
    def load_feedback(cls) -> str:
        """读取最近N轮反馈摘要，给Flash看上一轮因子Failed的原因。"""
        try:
            if not cls._FEEDBACK_FILE.exists():
                return ""
            history = json.loads(cls._FEEDBACK_FILE.read_text())
            if not history:
                return ""
            recent = history[-cls._MAX_ROUNDS_FEEDBACK:]
            total_rounds = len(recent)
            last = recent[-1]
            last_fail = last.get("fail_reasons", {})

            # ── 汇总失败原因 ──
            reasons = []
            if last_fail.get("syntax_error", 0) > 0:
                reasons.append(f"语法错误{last_fail['syntax_error']}个（已自动修复）")
            if last_fail.get("ic_too_low", 0) > 0:
                reasons.append(f"IC太低{last_fail['ic_too_low']}个")
            if last_fail.get("icir_too_low", 0) > 0:
                reasons.append(f"ICIR不稳定{last_fail['icir_too_low']}个")
            if last_fail.get("oos_decay", 0) > 0:
                reasons.append(f"OOS衰减{last_fail['oos_decay']}个")
            if last_fail.get("mp_failed", 0) > 0:
                reasons.append(f"多周期不通过{last_fail['mp_failed']}个")
            reasons_str = ", ".join(reasons) if reasons else "全部通过"

            # ── 计算近N轮趋势 ──
            all_produced = sum(
                r.get("strong_count", 0) - (history[max(0, i-1)].get("strong_count", 0) if i > 0 else 0)
                for i, r in enumerate(recent)
            )
            all_errors = sum(r.get("errors_fixed", 0) for r in recent)
            top_fail = ""
            # 找出最常见的失败原因
            _fail_counts: dict[str, int] = {}
            for r in recent:
                fr = r.get("fail_reasons", {})
                for k, v in fr.items():
                    if k != "syntax_error":
                        _fail_counts[k] = _fail_counts.get(k, 0) + v
            if _fail_counts:
                top_reason = max(_fail_counts, key=_fail_counts.get)
                top_fail = {
                    "ic_too_low": "IC太低（<0.02）是最常见淘汰原因，尝试更强信号方向",
                    "icir_too_low": "ICIR不稳定最常见，注意因子鲁棒性",
                    "oos_decay": "OOS衰减最常见，检查是否过拟合",
                    "mp_failed": "多周期不通过最常见，注意长期预测力",
                }.get(top_reason, "")

            lines = [
                f"\n【上一轮失败诊断】（共评估{last.get('evaluated',0)}个，入{last.get('strong_count',0)}强+{last.get('candidate_count',0)}候选）",
                f"  失败分布: {reasons_str}",
            ]
            if top_fail:
                lines.append(f"  ⚠ 近{total_rounds}轮趋势: {top_fail}")
            if all_produced == 0 and total_rounds >= 3:
                lines.append("  🚨 连续多轮无新因子入库，请彻底更换因子类型")
            elif all_produced == 0:
                lines.append("  ⚠ 近期无入库，请检查是否在同个方向反复尝试")
            lines.append("")
            return "\n".join(lines)
        except Exception:
            return ""

    def save_library(self) -> None:
        """Persist factor library to ~/.qlib/factor_library.json.

        Both strong and candidate tiers are saved (with tier field set).
        Merges with existing data on disk to preserve factors from past runs.
        """
        self._LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # 快照旧版因子库
        self._snapshot_before(self._LIBRARY_FILE)
        existing = self.load_library_from_disk()
        current_exprs = {f["expr"] for f in self._factor_library + self._factor_candidates}
        # Combine: current strong + current candidates + any prior factors not already present
        merged = (
            self._factor_library
            + self._factor_candidates
            + [f for f in existing if f["expr"] not in current_exprs]
        )
        # Sort: strong first, then candidates, both sorted by |IC|
        merged.sort(key=lambda x: (x.get("tier", "strong") != "strong", -abs(x.get("ic_mean", 0))))
        # ── 2. 结构去重：同表达式结构只保留IC最高的 ──────────────
        import re as _re2
        before = len(merged)
        _deduped: dict[str, dict] = {}
        _stem_count = 0
        for f in merged:
            stem = _re2.sub(r'\b\d{2,}\b', 'N', f["expr"])
            # 也归一化 1e-N 常数
            stem = _re2.sub(r'1e-\d+', '1e-M', stem)
            if stem in _deduped:
                existing_ic = abs(_deduped[stem].get("ic_mean", 0))
                current_ic = abs(f.get("ic_mean", 0))
                if current_ic > existing_ic:
                    _deduped[stem] = f
                _stem_count += 1
            else:
                _deduped[stem] = f
        merged = list(_deduped.values())
        removed = before - len(merged)
        if removed > 0:
            logger.info(f"[FactorAgent] 去重: 结构相同因子{removed}个（保留各结构最高IC版本），{before}→{len(merged)}")
        self._LIBRARY_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        strong = sum(1 for f in merged if f.get("tier", "strong") == "strong")
        logger.info(
            f"[FactorAgent] 因子库已保存: 强 {strong} + 候选 {len(merged)-strong} = {len(merged)} → {self._LIBRARY_FILE}"
        )

    @classmethod
    def load_library_from_disk(cls, tier: str = "all") -> list[dict]:
        """Load persisted factor library from disk.

        tier: 'strong' | 'candidate' | 'all' (default)
        Returns [] if not found.
        """
        try:
            if cls._LIBRARY_FILE.exists():
                data = json.loads(cls._LIBRARY_FILE.read_text())
                if not isinstance(data, list):
                    return []
                if tier == "strong":
                    return [f for f in data if f.get("tier", "strong") == "strong"]
                if tier == "candidate":
                    return [f for f in data if f.get("tier") == "candidate"]
                return data
        except Exception:
            pass
        return []

    # ── 诊断压缩摘要（AI 诊断 → 因子发现反馈） ────────────────────────

    @classmethod
    def save_diagnosis_compact(cls, diagnosis: dict) -> None:
        """保存诊断的压缩版本（只保留因子方向 + 关键建议，不膨胀）。"""
        try:
            compact = {
                "next_factor_direction": (diagnosis.get("next_factor_direction") or "").strip()[:200],
                "recommendations": [
                    r.get("action", "") for r in (diagnosis.get("recommendations") or [])
                    if r.get("priority") in ("高", "中")
                ][:3],
                "updated_at": datetime.today().strftime("%Y-%m-%d %H:%M"),
            }
            cls._DIAGNOSIS_COMPACT_FILE.parent.mkdir(parents=True, exist_ok=True)
            cls._DIAGNOSIS_COMPACT_FILE.write_text(json.dumps(compact, ensure_ascii=False))
            logger.info(f"[FactorAgent] 诊断压缩摘要已保存 → {cls._DIAGNOSIS_COMPACT_FILE}")
        except Exception as e:
            logger.warning(f"[FactorAgent] 保存诊断压缩摘要失败: {e}")

    @classmethod
    def load_diagnosis_compact(cls) -> str:
        """加载诊断压缩摘要，返回适合注入提示词的短文本（空串=无）。"""
        try:
            if cls._DIAGNOSIS_COMPACT_FILE.exists():
                d = json.loads(cls._DIAGNOSIS_COMPACT_FILE.read_text())
                parts = []
                if d.get("next_factor_direction"):
                    parts.append(f"上次诊断建议的研究方向：{d['next_factor_direction'][:150]}")
                if d.get("recommendations"):
                    rec_text = "；".join(d["recommendations"])
                    parts.append(f"建议关注：{rec_text[:100]}")
                if parts:
                    return "诊断反馈：" + "。".join(parts) + "。"
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    # IC evaluation
    # ------------------------------------------------------------------

    def _evaluate_factor_ic(
        self,
        expr: str,
        start: str = "2022-01-01",
        end: str = "2023-12-31",
        n_sample: int = 200,
        oos_split: float = 0.7,
    ) -> dict:
        """Compute cross-sectional IC at 3 horizons + out-of-sample IC.

        GKX 2020: a factor robust at multiple horizons (1d/5d/20d) is more reliable
        than one that fits a single holding period.
        McLean & Pontiff 2016: OOS IC ≥ IS IC × _IC_OOS_RATIO guards against data mining.
        Stores ic_series (1d) for correlation-dedup (Feng et al. 2020).

        Returns {ic_mean, icir, ic_std, n_days, oos_ic_mean,
                 ic_5d, ic_20d, multi_period_pass, ic_series, error?}.
        """
        _ZERO = {"ic_mean": 0.0, "icir": 0.0, "ic_std": 0.0, "n_days": 0,
                 "oos_ic_mean": 0.0, "ic_5d": 0.0, "ic_20d": 0.0,
                 "multi_period_pass": 0, "ic_series": [], "error": "skipped"}
        # 内存保护：可用内存不足2GB时跳过 D.features（防OOM）
        try:
            import psutil as _psutil_check
            _avail_gb = _psutil_check.virtual_memory().available / 1024**3
            if _avail_gb < 2.0:
                logger.warning(f"[FactorAgent] 可用内存 {_avail_gb:.1f}GB < 2GB，跳过表达式评估（防OOM）")
                return {**_ZERO, "error": f"low memory ({_avail_gb:.1f}GB)"}
        except ImportError:
            pass

        try:
            from qlib.data import D
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            codes = self._get_sample_codes(n_sample)

            # Wrap D.features in a thread with timeout to avoid hanging on complex expressions
            _FEATURES_TIMEOUT = 120  # seconds per factor (Original:60)
            def _fetch():
                return D.features(
                    codes,
                    fields=[
                        expr,
                        "Ref($close,-1)/$close-1",   # 1-day
                        "Ref($close,-5)/$close-1",   # 5-day
                        "Ref($close,-20)/$close-1",  # 20-day
                    ],
                    start_time=start,
                    end_time=end,
                    freq="day",
                )

            _ex = ThreadPoolExecutor(max_workers=1)
            _fut = _ex.submit(_fetch)
            _ex.shutdown(wait=False)  # don't block on shutdown; let thread run detached
            try:
                data = _fut.result(timeout=_FEATURES_TIMEOUT)
            except FuturesTimeout:
                logger.warning(f"[FactorAgent] D.features 超时({_FEATURES_TIMEOUT}s)，跳过表达式: {expr[:60]}")
                return {**_ZERO, "error": f"timeout after {_FEATURES_TIMEOUT}s"}

            if data is None or data.empty:
                return _ZERO

            data.columns = ["factor", "fwd_1d", "fwd_5d", "fwd_20d"]
            data = data.dropna(subset=["factor", "fwd_1d"])

            if len(data) < 100:
                return _ZERO

            def _cross_ic(df, fwd_col: str):
                series = df.groupby(level="datetime").apply(
                    lambda g: g["factor"].corr(g[fwd_col])
                ).dropna()
                if len(series) == 0:
                    return None, 0.0, 0.0, 0.0
                m = float(series.mean())
                s = float(series.std())
                ir = m / s if s > 1e-8 else 0.0
                return series, m, s, ir

            ic_ser, ic_mean, ic_std, icir = _cross_ic(data, "fwd_1d")
            if ic_ser is None:
                return _ZERO

            n_days = int(len(ic_ser))

            # OOS split on 1-day IC series
            split_idx = int(n_days * oos_split)
            oos_ic_mean = float(ic_ser.iloc[split_idx:].mean()) if (n_days - split_idx) > 10 else 0.0

            # 5-day and 20-day IC (single mean, fast)
            data_5d = data.dropna(subset=["fwd_5d"])
            data_20d = data.dropna(subset=["fwd_20d"])
            _, ic_5d, _, _ = _cross_ic(data_5d, "fwd_5d")
            _, ic_20d, _, _ = _cross_ic(data_20d, "fwd_20d")

            # GKX multi-period check: count horizons where IC > small positive threshold
            multi_pass = sum([
                abs(ic_mean) > 0.005,
                abs(ic_5d)   > 0.005,
                abs(ic_20d)  > 0.005,
            ])

            # ── L1 组合回测：多空组合验证（IC 高不代表能赚钱） ─────────────
            def _portfolio_bt(df):
                """Simple long-short portfolio backtest on daily data.

                Each day: long top 10% by factor, short bottom 10%.
                Equal-weighted, rebalanced daily. Returns (sharpe, max_dd).
                """
                import numpy as np
                _rets = []
                for _dt, _group in df.groupby(level="datetime"):
                    _group = _group.sort_values("factor", ascending=False)
                    _n = len(_group)
                    _long = _group.iloc[:_n // 10]
                    _short = _group.iloc[-_n // 10:]
                    if _long.empty or _short.empty:
                        continue
                    _ret = _long["fwd_1d"].mean() - _short["fwd_1d"].mean()
                    _rets.append(_ret)
                if len(_rets) < 20:
                    return 0.0, 0.0
                _arr = np.array(_rets, dtype=float)
                _sharpe = float(_arr.mean() / _arr.std() * np.sqrt(252)) if _arr.std() > 1e-8 else 0.0
                _peak = np.maximum.accumulate(1 + np.cumsum(_arr))
                _dd = (1 + np.cumsum(_arr)) / _peak - 1
                _mdd = float(np.min(_dd))
                return round(_sharpe, 3), round(_mdd, 4)
            bt_sharpe, bt_mdd = _portfolio_bt(data)

            return {
                "ic_mean":          round(ic_mean, 4),
                "icir":             round(icir, 4),
                "ic_std":           round(ic_std, 4),
                "n_days":           n_days,
                "oos_ic_mean":      round(oos_ic_mean, 4),
                "ic_5d":            round(ic_5d, 4),
                "ic_20d":           round(ic_20d, 4),
                "multi_period_pass": multi_pass,
                "bt_sharpe":        bt_sharpe,
                "bt_max_dd":        bt_mdd,
                "ic_series":        [round(v, 5) for v in ic_ser.tolist()],
            }
        except Exception as exc:
            logger.warning(f"[FactorAgent] IC 计算失败 ({expr[:40]}): {exc}")
            return {**_ZERO, "error": str(exc)}

    def _get_sample_codes(self, n: int = 300) -> list[str]:
        """Return a random sample of exchange-prefixed stock codes."""
        if not self._qlib_codes:
            feat_dir = self.data_dir / "features"
            if feat_dir.exists():
                self._qlib_codes = [
                    d.name for d in feat_dir.iterdir()
                    if d.is_dir() and len(d.name) >= 8
                ]
        if not self._qlib_codes:
            return []
        return random.sample(self._qlib_codes, min(n, len(self._qlib_codes)))

    # ------------------------------------------------------------------
    # Qlib init
    # ------------------------------------------------------------------

    def _init_qlib(self) -> bool:
        if self._initialized:
            return True
        try:
            import qlib
            from qlib.config import REG_CN
            qlib.init(provider_uri=str(self.data_dir), region=REG_CN)
            self._initialized = True
            logger.info(f"[FactorAgent] Qlib 初始化完成: {self.data_dir}")
            return True
        except Exception as exc:
            logger.error(f"[FactorAgent] Qlib 初始化失败: {exc}")
            return False

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, temperature: float = 0.7, model_hint: str = "auto") -> str:
        """Direct LLM API call (no LangChain overhead). 2026-08-28 迁移智谱GLM。

        model_hint:
          "pro"   — 只用 GLM-5.3（方向设定/审计）
          "flash" — 只用 GLM-5.3-Flash（批量挖掘）
          "auto"  — 先试 pro，失败回退 flash（默认）
        """
        import requests

        api_key = (
            os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("TRADINGAGENTS_DEEPSEEK_API_KEY")
            or ""
        )
        base_url = (
            os.getenv("TRADINGAGENTS_LLM_BACKEND_URL")
            or os.getenv("LLM_BACKEND_URL")
            or os.getenv("backend_url")
            or "https://open.bigmodel.cn/api/coding/paas/v4"
        ).rstrip("/")

        quick = (os.getenv("QUICK_THINK_LLM") or os.getenv("quick_think_llm") or "glm-5.3-flash")
        deep = (os.getenv("DEEP_THINK_LLM") or os.getenv("deep_think_llm") or "glm-5.3")
        # 用户指定用 pro 因子分析，超时或失败时自动回退到 flash
        if model_hint == "pro":
            models_to_try = [deep]
        elif model_hint == "flash":
            models_to_try = [quick]
        else:
            models_to_try = [deep] if deep == quick else [deep, quick]

        # 末段已是版本号(/v1|/v4)则只补 /chat/completions（bigmodel 是 /api/paas/v4）
        last_seg = base_url.split("/")[-1]
        url = (f"{base_url}/chat/completions" if last_seg.startswith("v") and last_seg[1:].isdigit()
               else f"{base_url}/v1/chat/completions")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        proxies = None
        if "bigmodel" not in base_url:  # bigmodel 国内直连不走代理
            https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
            if https_proxy:
                proxies = {"https": https_proxy, "http": https_proxy}

        last_error = ""
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 16384,
            }
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     proxies=proxies, timeout=120)
                resp.raise_for_status()
                resp_data = resp.json()
                content = resp_data["choices"][0]["message"]["content"]
                if not content or content.strip() == "":
                    logger.warning(f"[FactorAgent] LLM返回空内容 (model={model}, finish_reason={resp_data['choices'][0].get('finish_reason','?')})")
                    if model != models_to_try[-1]:
                        continue
                    return f"[分析失败: LLM返回空内容]"
                if model != models_to_try[0]:
                    logger.info(f"[FactorAgent] 模型回退成功: {model} → 正常返回")
                return content
            except Exception as e:
                detail = f" ({type(e).__name__}: {e})"
                _resp = locals().get("resp")
                if _resp is not None:
                    try:
                        detail = f" (HTTP {_resp.status_code} {_resp.text[:200]})"
                    except Exception:
                        pass
                last_error = f"{model} 调用失败{detail}"
                logger.warning(f"[FactorAgent] {last_error}")
                continue

        logger.error(f"[FactorAgent] 所有模型均失败: {last_error}")
        return f"[分析失败: {last_error}]"

    @staticmethod
    def _parse_json_from_llm(content: str) -> dict:
        """Extract JSON object from LLM response (handles markdown code fences)."""
        # Strip markdown fences
        content = re.sub(r"```(?:json)?", "", content).strip()
        # Find first {...} block
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            return json.loads(m.group())
        return json.loads(content)

    @staticmethod
    def _summarize_history(history: list[dict]) -> str:
        """Produce a concise text summary of past iterations for the LLM.

        Only the last 2 iterations are included in full; earlier ones are
        summarized as "第N轮: X/X通过/淘汰" to keep prompt length under control.
        Only top accepted and bottom rejected factors are shown per iteration.
        """
        if not history:
            return ""
        lines = []
        # Only show last 2 iterations in detail; older ones get a one-liner
        recent_threshold = max(0, len(history) - 2)
        for i, h in enumerate(history):
            it = h["iteration"]
            results = h.get("results", [])
            accepted = [
                r for r in results
                if (r.get("ic_mean", 0) >= _IC_GOOD
                    and r.get("icir", 0) >= _ICIR_MIN
                    and r.get("oos_ic_mean", 0) >= _IC_GOOD * _IC_OOS_RATIO)
            ]
            rejected = [r for r in results if r not in accepted]

            if i < recent_threshold:
                # Older iteration: one-liner
                lines.append(
                    f"第{it}轮: {len(accepted)}通过 / {len(rejected)}淘汰"
                )
                continue

            # Recent iteration: full detail, but cap to 3 each
            if accepted:
                lines.append(f"第{it}轮通过（IC+ICIR+OOS均达标）:")
                for r in accepted[:3]:
                    lines.append(
                        f"  IC={r['ic_mean']:.4f} ICIR={r['icir']:.4f} OOS={r.get('oos_ic_mean',0):.4f}: {r['expr'][:65]}"
                    )
                if len(accepted) > 3:
                    lines.append(f"  ...还有 {len(accepted)-3} 个通过")
            if rejected:
                cap = 3 if accepted else 5
                lines.append(f"第{it}轮未通过（展示前{cap}个）:")
                for r in rejected[:cap]:
                    lines.append(f"  IC={r['ic_mean']:.4f} ICIR={r.get('icir',0):.4f}: {r['expr'][:50]}")
                if len(rejected) > cap:
                    lines.append(f"  ...还有 {len(rejected)-cap} 个未通过")
        return "\n".join(lines)
