"""集中管理因子工厂的文件路径（env FF_HOME 可整体重定向，默认 ~/.factor_factory）。"""
import os
from pathlib import Path

FF_HOME = Path(os.getenv("FF_HOME", str(Path.home() / ".factor_factory")))
FF_HOME.mkdir(parents=True, exist_ok=True)

LIBRARY_FILE = FF_HOME / "factor_library.json"
LOG_FILE = FF_HOME / "factor_factory.log"
PID_FILE = FF_HOME / "factor_factory.pid"
STOP_FILE = FF_HOME / "factor_factory.stop"
PAUSE_FILE = FF_HOME / "factor_factory.pause"
DIRECTION_FILE = FF_HOME / "factor_factory.direction.txt"
PROGRESS_FILE = FF_HOME / "factor_progress.json"
DECAY_REPORT_FILE = FF_HOME / "factor_decay_report.json"

# qlib A股数据目录（IC 评估用，用户自备；兼容 qlib 生态默认位置）
DATA_DIR = Path(os.getenv("FF_DATA_DIR", str(Path.home() / ".qlib" / "cn_data")))
