# Factor Factory

[中文](README.md) | **English**

LLM-driven alpha factor factory for China A-shares. Inspired by Microsoft [RD-Agent](https://github.com/microsoft/RD-Agent) (NeurIPS 2025), but built with the opposite goal: **not a heavyweight research framework — a ready-to-run factor production machine**.

```
LLM proposes Qlib factor expressions → real market data validates via IC → auto-tiered into library → diagnostics feed the next round → runs around the clock
```

## Why this one

| | RD-Agent-Quant | **Factor Factory** |
|---|---|---|
| Positioning | Research framework (heavy) | Production machine (light): core loop is a few hundred lines |
| Operation | Manual experiments | **Always-on daemon** with file-based control protocol (stop/pause/direction) |
| Acceptance | Paper-grade evaluation | Hard gates: IC / ICIR / OOS auto-tiering (strong/candidate) |
| Ecosystem | Generic | China A-share native: qlib expressions, CN market conventions, Chinese factor diagnostics |

## Quick Start (~10 min)

```bash
# 0) Prerequisite: qlib-format China A-share daily bars (Microsoft Qlib dump format,
#    compatible with ~/.qlib/cn_data)

# 1) Install
pip install -e .

# 2) Configure LLM (any OpenAI-compatible endpoint)
cp .env.example .env    # fill LLM_API_KEY / LLM_BASE_URL / model names
set -a; source .env; set +a

# 3) Seed the library (optional; 3 public qlib-Alpha158 factors included)
mkdir -p ~/.factor_factory
cp seeds/factor_library.seeds.json ~/.factor_factory/factor_library.json

# 4) Run one full R&D loop (or start the always-on daemon)
python -c "
from factor_factory.agent import FactorAgent
fa = FactorAgent()
fa.run_research_loop(n_iter=1, factors_per_iter=5)
"
cat ~/.factor_factory/factor_library.json   # new factors are in
```

## Always-on daemon (recommended)

```bash
./scripts/factor_factory.sh start     # background resident: 15 factors/round, max 16 rounds/day
./scripts/factor_factory.sh status
./scripts/factor_factory.sh stop      # file protocol: writes a stop file
tail -f ~/.factor_factory/factor_factory.log
```

All control is file-based signals — no sockets, no database: `factor_factory.stop` / `.pause` / `.direction.txt` (steer the mining direction, overriding LLM auto-diagnosis).

## IC tiering gates (defaults, tunable)

| tier | condition | meaning |
|---|---|---|
| strong | IC≥0.02, ICIR≥0.3, OOS≥70%, ≥2 predictive horizons | ready for model-training candidates |
| candidate | IC≥0.015, ICIR≥0.2 (loosened) | watchlist |

`IC = cross-sectional Pearson corr(factor value, next-day return)`, evaluated on real A-share daily bars (qlib) with rolling windows to prevent look-ahead.

## Design notes

- **LLM proposes, data decides**: the LLM only generates hypotheses and interprets results; admission is gated by hard IC thresholds — the LLM cannot argue with statistics.
- **Diagnostic feedback** ("inference as gradient"): quantified diagnostics of failed factors are injected into the next round's prompt — converges faster than random search.
- **File protocol**: control signals are files, not sockets/DB — `rm` to stop, `cat` to configure, crash-safe.
- **Pre-fetch direct calls**: data acquisition is deterministic Python code; the LLM never touches the data path, so hallucinations cannot contaminate IC.

## Disclaimer

For quantitative research and educational purposes only. Not investment advice. Factors decay.

## License

MIT
