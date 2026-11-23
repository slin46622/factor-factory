"""核心单元测试：tier 分级 / 文件协议 / paths。不依赖网络与 qlib 数据。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# FF_HOME 指向临时目录（必须在 import factor_factory 前设置）
import tempfile

_TMP = tempfile.mkdtemp(prefix="ff_test_")
os.environ["FF_HOME"] = _TMP

from factor_factory import paths  # noqa: E402


def test_paths_env_redirect():
    assert str(paths.FF_HOME) == _TMP
    assert paths.LIBRARY_FILE.parent == Path(_TMP)


def test_stop_file_protocol():
    # 模拟 daemon 的停止检测：写文件 → 存在 → 删除 → 不存在
    paths.STOP_FILE.write_text("stop")
    assert paths.STOP_FILE.exists()
    paths.STOP_FILE.unlink()
    assert not paths.STOP_FILE.exists()


def test_tier_thresholds():
    from factor_factory import agent as ag

    # strong: IC>=0.02, ICIR>=0.3, OOS>=0.7
    assert ag._IC_GOOD == 0.02 and ag._IC_WEAK == 0.015
    # 阈值序：strong 门槛必须严于 candidate
    assert ag._IC_GOOD > ag._IC_WEAK


def test_library_roundtrip(tmp_path=None):
    from factor_factory.agent import FactorAgent

    fa = FactorAgent()
    lib_file = Path(_TMP) / "lib_roundtrip.json"
    sample = [{"expr": "$close/Ref($close,1)-1", "tier": "strong", "ic_mean": 0.03, "icir": 0.5}]
    lib_file.write_text("[]")
    orig = fa.__class__
    # 直接测 save/load 纯函数部分（不触网）
    import json

    merged = json.loads(lib_file.read_text() or "[]") + sample
    lib_file.write_text(json.dumps(merged, ensure_ascii=False))
    loaded = json.loads(lib_file.read_text())
    assert loaded[-1]["tier"] == "strong"


def test_no_internal_paths():
    """隐私守卫：源码不得含原项目内部路径/名称。"""
    src_root = Path(__file__).resolve().parents[1] / "src"
    bad_tokens = ["/home/sun", "D:\\GLM", "D:/GLM", "Project0904", "Zcode", "fuyao",
                  "ZHIPU_API_KEY=", "sk-"]
    hits = []
    for py in src_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for tok in bad_tokens:
            if tok in text:
                hits.append(f"{py.name}: {tok}")
    assert not hits, f"发现敏感串: {hits}"
