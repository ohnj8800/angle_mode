from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SCRIPT_PATTERN = re.compile(r"^(\d{2})_.*\.py$")


def _discover_scripts() -> list[tuple[int, Path]]:
    discovered = []
    for path in SCRIPT_ROOT.glob("[0-9][0-9]_*.py"):
        match = SCRIPT_PATTERN.match(path.name)
        if match:
            discovered.append((int(match.group(1)), path))
    return sorted(discovered, key=lambda item: (item[0], item[1].name))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依編號順序執行FBG角度與VMD完整分析流程。"
    )
    parser.add_argument(
        "--from-step", type=int, default=1,
        help="從哪一步開始執行，預設為1。",
    )
    parser.add_argument(
        "--to-step", type=int, default=99,
        help="執行到哪一步，預設執行所有現有編號腳本。",
    )
    parser.add_argument(
        "--skip-step", type=int, action="append", default=[],
        help="略過指定步驟；可重複使用，例如 --skip-step 4。",
    )
    return parser.parse_args()


def _run_script(step: int, path: Path) -> float:
    print("\n" + "=" * 72, flush=True)
    print(f"開始執行第{step:02d}步：{path.name}", flush=True)
    print("=" * 72, flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(path)], cwd=PROJECT_ROOT, check=False
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"第{step:02d}步執行失敗：{path.name}；"
            f"return code={completed.returncode}"
        )
    print(f"第{step:02d}步完成，耗時 {elapsed:.2f} 秒", flush=True)
    return elapsed


def main() -> None:
    arguments = _parse_arguments()
    skip_steps = set(arguments.skip_step)
    scripts = [
        (step, path)
        for step, path in _discover_scripts()
        if arguments.from_step <= step <= arguments.to_step
        and step not in skip_steps
    ]
    if not scripts:
        raise SystemExit("指定範圍內沒有可執行的分析腳本。")

    print("=" * 72)
    print("FBG／VMD完整分析流程")
    print("即將執行：")
    for step, path in scripts:
        print(f"  {step:02d}  {path.name}")
    print("=" * 72, flush=True)

    total_started = time.perf_counter()
    timings = []
    try:
        for step, path in scripts:
            timings.append((step, path.name, _run_script(step, path)))
    except RuntimeError as error:
        print(f"\n流程已停止：{error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error

    print("\n" + "=" * 72)
    print("全部指定步驟執行完成")
    for step, name, elapsed in timings:
        print(f"  {step:02d}  {elapsed:8.2f} 秒  {name}")
    print(f"總耗時：{time.perf_counter() - total_started:.2f} 秒")
    print(f"結果位置：{PROJECT_ROOT / 'results'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
