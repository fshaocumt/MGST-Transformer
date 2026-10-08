"""Batch orchestrator for the full experimental protocol.
Repeatedly calls run_experiments for every configuration: hyper-parameter
search, main model (three fusion modes x two years x five seeds), gate-weight
export, baselines and ablations.
"""

import argparse
import os
import subprocess
import sys
import time

import run_experiments as RE

BUILD = "2026-10-05a"

DATA18 = "/root/autodl-fs/preprocess_result"
DATA24 = "/root/autodl-fs/preprocess_result24年"
TAG18 = "18年"
TAG24 = "24年"
OUT18 = "results2018"
OUT24 = "results2024"
SEEDS = ["42", "7", "2024", "123", "2026"]
SEEDS_ABLATION = ["42"]
BATCH = 64
SEARCH_EPOCHS = 30
SEARCH_ROUNDS = 1

DATASETS = [(DATA18, TAG18, OUT18), (DATA24, TAG24, OUT24)]

FAILURES = []
ALLOW_DUP = False
VERBOSE = False

def _call(dry, argv):
    line = " ".join(argv)
    if dry:
        print("[dry-run] " + line)
        return
    print("\n>>> " + line, flush=True)
    try:
        RE.main(argv)
    except Exception as e:
        FAILURES.append(line)
        print(f"[ERROR] {line}\n        {type(e).__name__}: {e}", flush=True)

def _base(dir_, tag, out):
    argv = ["--data-dir", dir_, "--out", out, "--batch-size", str(BATCH)]
    if tag not in (None, ""):
        argv += ["--tag", tag]
    if VERBOSE:
        argv += ["--verbose"]
    return argv

def _search_args():
    return ["--search-epochs", str(SEARCH_EPOCHS),
            "--search-rounds", str(SEARCH_ROUNDS)]

def search(dry=False, fusion="node"):
    for dir_, tag, out in DATASETS:
        _call(dry, _base(dir_, tag, out) + [
            "--model", "mgst", "--fusion", fusion,
            "--search", "--search-only"])

def main(dry=False, fusions=("global", "sample", "node")):
    order = ["node"] + [f for f in fusions if f != "node"]
    for fusion in order:
        for dir_, tag, out in DATASETS:
            argv = _base(dir_, tag, out) + [
                "--model", "mgst", "--fusion", fusion, "--seeds"] + SEEDS
            if fusion == "node":
                argv += (["--search"] + _search_args()
                         + ["--dump-alpha", "--save-preds"])
            else:
                argv += ["--inherit", "mgst_node"]
            _call(dry, argv)

def alpha(dry=False):
    for dir_, tag, out in DATASETS:
        _call(dry, _base(dir_, tag, out) + [
            "--model", "mgst", "--fusion", "node",
            "--seeds", "42", "--dump-alpha", "--inherit", "mgst_node"])

BASELINES = ["naive", "lstm", "gru", "tcn", "transformer", "stid",
             "stgcn", "gwnet", "astgcn", "staeformer", "dcrnn"]

def baselines(dry=False, models=None):
    for m in (models or BASELINES):
        for dir_, tag, out in DATASETS:
            argv = _base(dir_, tag, out) + ["--model", m]
            if m == "naive":
                argv += ["--seeds", "42"]
            else:
                argv += ["--search", "--seeds"] + SEEDS + _search_args()
            _call(dry, argv)

ABLATION = {
    "ablate_wo_distance":   ["--graphs", "flow_corr", "speed_corr", "region"],
    "ablate_wo_flowcorr":   ["--graphs", "distance", "speed_corr", "region"],
    "ablate_wo_speedcorr":  ["--graphs", "distance", "flow_corr", "region"],
    "ablate_wo_region":     ["--graphs", "distance", "flow_corr", "speed_corr"],
    "ablate_wo_twocorr":    ["--graphs", "distance", "region"],
    "ablate_wo_speedfeat":  ["--mask", "no_speed"],
    "ablate_wo_timefeat":   ["--mask", "no_time"],
}

def _other_ablation_procs():
    try:
        text = subprocess.run(["ps", "-ef"], capture_output=True,
                              text=True, check=True).stdout
    except FileNotFoundError:
        return []
    me = {os.getpid(), os.getppid()}
    return [(pid, cmd) for pid, cmd in _parse_ps(text) if pid not in me]

def _require_inherit(name="mgst_node"):
    missing = [out for _, _, out in DATASETS
               if not os.path.exists(os.path.join(out, f"{name}_best.json"))]
    if missing:
        raise SystemExit(
            f"[FATAL] 缺少 {' / '.join(missing)} 里的 {name}_best.json，"
            f"消融无法继承超参（不继承的话删图就不是单变量对照，结果作废）。\n"
            f"        先补跑：python run_all.py --step main\n"
            f"        （main 只会跑 node 的搜索 + 三模式，已完成的会自动跳过）")

def ablation(dry=False, variants=None):
    _require_inherit()
    others = _other_ablation_procs()
    if others and not ALLOW_DUP:
        raise SystemExit(
            "[FATAL] 已经有别的消融进程在跑，再起一组会抢着写同一个 _overall.csv：\n"
            + "\n".join(f"         pid={p}  {c[:100]}" for p, c in others)
            + "\n\n        先停掉它们：python run_all.py --step stop\n"
            + "        确认干净后再跑。确实要并发（比如已用 --chunk 分好片）就加 --allow-dup。")
    if others:
        print(f"[warn] 检测到 {len(others)} 个其它消融进程（--allow-dup 已开启）", flush=True)
    for name, extra in (ABLATION.items() if variants is None else variants):
        for dir_, tag, out in DATASETS:
            _call(dry, _base(dir_, tag, out) + [
                "--model", "mgst", "--fusion", "node", "--name", name,
                "--inherit", "mgst_node"]
                + extra + ["--seeds"] + SEEDS_ABLATION)

def _spawn(year, chunk=None, stagger=20):
    argv = [sys.executable, os.path.abspath(__file__),
            "--step", "ablation", "--year", year]
    if chunk:
        argv += ["--chunk", chunk]
    log = f"ablation_{year}" + (f"_{chunk.replace('/', '-')}" if chunk else "") + ".log"
    f = open(log, "w", encoding="utf-8")
    p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"[spawn] pid={p.pid}  year={year} {chunk or ''}  -> {log}", flush=True)
    time.sleep(stagger)
    return p

def ablation_par(dry=False, years=("18", "24"), parts=1):
    chunks = [None] if parts <= 1 else [f"{i}/{parts}" for i in range(parts)]
    procs = []
    for y in years:
        for c in chunks:
            if dry:
                cmd = " ".join([sys.executable, "run_all.py", "--step", "ablation",
                                "--year", y] + (["--chunk", c] if c else []))
                print("[dry-run] " + cmd)
                continue
            procs.append(_spawn(y, c))
    if procs:
        print(f"\n[run_all] 已后台启动 {len(procs)} 个进程。查看进度：")
        print("          tail -f ablation_18.log   /   tail -f ablation_24.log")
        print("          或者：python run_all.py --step progress")
    return procs

def _scan():
    done, todo = [], []
    for dir_, tag, out in DATASETS:
        for name in ABLATION:
            for s in SEEDS_ABLATION:
                path = os.path.join(out, f"{name}_seed{s}_overall.csv")
                (done if os.path.exists(path) else todo).append(path)
    return done, todo

def _per_out(done):
    cnt = {out: 0 for _, _, out in DATASETS}
    total = {out: len(ABLATION) * len(SEEDS_ABLATION) for _, _, out in DATASETS}
    for p in done:
        for _, _, out in DATASETS:
            if p.startswith(out + os.sep) or p.startswith(out + "/"):
                cnt[out] += 1
                break
    return cnt, total

def _last_line(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        return lines[-1][:160] if lines else "(空)"
    except FileNotFoundError:
        return "(无日志)"

def progress(dry=False, verbose=False):
    done, todo = _scan()
    total = len(done) + len(todo)
    cnt, nc = _per_out(done)
    print(f"[progress] 消融 {len(done)}/{total} 个 seed 已完成"
          + ("  —— 全部跑完，可以汇总 *_all_seeds.csv 了" if not todo else ""))
    for _, _, out in DATASETS:
        print(f"           {out}: {cnt[out]}/{nc[out]}")
    if verbose:
        for p in todo:
            print("  [缺] " + p)

def _parse_ps(text):
    hits = []
    for line in text.splitlines():
        if "run_all.py" not in line or "--step ablation" not in line:
            continue
        if "grep" in line or "ps -ef" in line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 2:
            continue
        try:
            hits.append((int(parts[1]), line.strip()))
        except ValueError:
            continue
    return hits

def ps(dry=False):
    try:
        text = subprocess.run(["ps", "-ef"], capture_output=True,
                              text=True, check=True).stdout
    except FileNotFoundError:
        print("[ps] 这台机器没有 ps 命令（Windows？），请在服务器上执行")
        return []
    hits = _parse_ps(text)
    if not hits:
        print("[ps] 没有正在跑的消融进程")
    for pid, cmd in hits:
        print(f"  pid={pid}  {cmd[:120]}")
    return hits

def stop(force=False):
    import signal
    hits = ps()
    if not hits:
        return
    sig, name = signal.SIGTERM, "SIGTERM"
    if force:
        sig = getattr(signal, "SIGKILL", None)
        if sig is None:
            print("[stop] 当前平台没有 SIGKILL，退回 SIGTERM")
            sig = signal.SIGTERM
        else:
            name = "SIGKILL"
    for pid, _ in hits:
        try:
            os.kill(pid, sig)
            print(f"[stop] 已发送 {name} -> {pid}")
        except ProcessLookupError:
            print(f"[stop] {pid} 已经不在了")
    print("[stop] 等几秒后确认：")
    print("          nvidia-smi            (应该看不到 python 占显存了)")
    print("          python run_all.py --step ps")
    print("          残留的 DataLoader worker：ps -ef | grep python | grep -v grep")

def watch(interval=300, rounds=None, heartbeat=6):
    import datetime
    seen = set()
    n = 0
    t0 = time.time()
    print(f"[watch] 每 {interval}s 检查一次，有新结果就刷新；Ctrl+C 退出", flush=True)
    try:
        while rounds is None or n < rounds:
            done, todo = _scan()
            total = len(done) + len(todo)
            new = [p for p in done if p not in seen]
            if new:
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"\n[{stamp}] 新完成 {len(new)} 个（累计 {len(done)}/{total}）：", flush=True)
                for p in new[:10]:
                    print("   + " + p, flush=True)
                if len(new) > 10:
                    print(f"   ...（还有 {len(new) - 10} 个，用 progress 看全量）", flush=True)
                seen.update(new)
            elif n % heartbeat == 0:
                el = int(time.time() - t0) // 60
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                rate = f"{len(done) / max(el / 60, 1e-9):.1f} 个/小时" if done and el else "—"
                print(f"[{stamp}] {len(done)}/{total}，已盯 {el} 分钟，速率 {rate}", flush=True)
                for log in ("ablation_18.log", "ablation_24.log"):
                    print(f"           {log}: {_last_line(log)}", flush=True)
            n += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch] 已停止（训练进程不受影响，仍在后台跑）")

def core(dry=False):
    main(dry)
    alpha(dry)
    ablation(dry)

STEPS = {"search": search, "core": core, "main": main, "alpha": alpha,
         "baselines": baselines, "ablation": ablation,
         "ablation_par": ablation_par, "progress": progress, "watch": watch,
         "ps": ps, "stop": stop}

if __name__ == "__main__":
    ap = argparse.ArgumentParser("run all MGST experiments")
    ap.add_argument("--step", choices=list(STEPS) + ["all"], default="core",
                    help="默认值是 core（主模型 + 门控图 + 消融，不含基线）。"
                         "在 Jupyter 里直接点「运行」时不会传任何命令行参数，"
                         "走的正是这个默认值；要复现基线请显式写 --step all。")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的调用")
    ap.add_argument("--year", choices=["18", "24", "all"], default="all",
                    help="只跑某一年。开两个进程（--year 18 / --year 24）可以"
                         "把两年并行起来，显存够的话总时间接近减半")
    ap.add_argument("--chunk", default=None, metavar="K/N",
                    help="只跑第 K 份（共 N 份）消融变体，配合多进程使用；"
                         "切分方式为交错（K=0 取第 0, N, 2N... 个变体），各份工作量均衡")
    ap.add_argument("--parts", type=int, default=1,
                    help="--step ablation_par 时，每年再切成几份进程（默认 1，"
                         "即总共 2 个进程；改成 2 就是 4 个进程）")
    ap.add_argument("--interval", type=int, default=300,
                    help="--step watch 的轮询间隔，单位秒（默认 300）")
    ap.add_argument("--rounds", type=int, default=None,
                    help="--step watch 最多轮询多少轮；不填就一直盯到 Ctrl+C")
    ap.add_argument("--verbose", action="store_true",
                    help="两用：① --step progress 时逐个列出还缺的文件；"
                         "② 训练时把 --verbose 透传给 run_experiments，逐 epoch 打印损失，"
                         "这样在 Jupyter/终端里能实时看到进度（日志会变长）")
    ap.add_argument("--force", action="store_true",
                    help="--step stop 时用 SIGKILL 强杀（默认 SIGTERM）")
    ap.add_argument("--ablation-seeds", nargs="+", default=None, metavar="S",
                    help="覆盖消融用的 seed 列表，例如 --ablation-seeds 42 7 2024 "
                         "（默认只有 42；种子越多时间线性增长）")
    ap.add_argument("--allow-dup", action="store_true",
                    help="即使检测到已有消融进程在跑也继续（配合 --chunk 分片并行时用）")
    a, _ = ap.parse_known_args()

    if a.ablation_seeds:
        SEEDS_ABLATION[:] = [str(s) for s in a.ablation_seeds]
        print(f"[run_all] 消融 seeds = {SEEDS_ABLATION}", flush=True)
    ALLOW_DUP = a.allow_dup
    VERBOSE = a.verbose

    if a.year != "all":
        DATASETS[:] = [DATASETS[0]] if a.year == "18" else [DATASETS[1]]
        print(f"[run_all] only {DATASETS[0][2]}", flush=True)

    if a.chunk:
        k, n = (int(x) for x in a.chunk.split("/"))
        variants = list(ABLATION.items())[k::n]
        print(f"[run_all] chunk {k}/{n} -> {[v[0] for v in variants]}", flush=True)
    else:
        variants = None
    print(f"[build] run_all.py {BUILD}  "
          f"(run_experiments.py {RE.BUILD})", flush=True)
    if a.step == "all":
        core(a.dry_run)
        baselines(a.dry_run)
    elif a.step == "ablation_par":
        ablation_par(a.dry_run, parts=a.parts)
    elif a.step == "ablation":
        ablation(a.dry_run, variants)
    elif a.step == "progress":
        progress(a.dry_run, verbose=a.verbose)
    elif a.step == "watch":
        watch(interval=a.interval, rounds=a.rounds)
    elif a.step == "stop":
        stop(force=a.force)
    elif a.step == "ps":
        ps()
    else:
        STEPS[a.step](a.dry_run)

    if FAILURES:
        print(f"\n[run_all] {len(FAILURES)} 个配置失败：", flush=True)
        for line in FAILURES:
            print("  " + line, flush=True)
        raise SystemExit(1)
