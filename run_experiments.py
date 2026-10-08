"""Single-configuration experiment runner: one model, one fusion mode, one dataset.
Supports coordinate-descent hyper-parameter search, multi-seed training,
ablations (graph / feature removal) and metric export.
"""

import argparse
import json
import os
import time

BUILD = "2026-10-02e"

import numpy as np
import pandas as pd
import torch

from mgst.baselines_stgnn import ASTGCN, DCRNN, STAEformer, STGCN, GraphWaveNet
from mgst.baselines_temporal import STID, PerNodeTemporal, naive_metrics
from mgst.common import (count_parameters, create_dataloaders, evaluate_model,
                         get_device, load_preprocessed, save_run, set_seed,
                         train_model)
from mgst.model import MGSTTransformer

ALL_GRAPHS = ["distance", "flow_corr", "speed_corr", "region"]

# Candidate grid for the coordinate-descent hyper-parameter search (section IV).
SEARCH_SPACE = {
    "mgst":        {"d_model": [32, 64, 128], "gnn_layers": [1, 2, 3],
                    "heads": [2, 4, 8], "trans_layers": [1, 2, 3],
                    "dropout": [0.0, 0.1, 0.2, 0.3], "lr": [1e-3, 5e-4]},
    "lstm":        {"hidden": [64, 128, 256], "num_layers": [1, 2, 3],
                    "dropout": [0.0, 0.1, 0.2]},
    "gru":         {"hidden": [64, 128, 256], "num_layers": [1, 2, 3],
                    "dropout": [0.0, 0.1, 0.2]},
    "tcn":         {"hidden": [64, 128, 256], "num_layers": [2, 3, 4],
                    "dropout": [0.0, 0.1, 0.2]},
    "transformer": {"hidden": [64, 128, 256], "num_layers": [1, 2, 3],
                    "heads": [2, 4, 8], "dropout": [0.0, 0.1, 0.2]},
    "stid":        {"hidden": [64, 128, 256], "num_layers": [2, 3, 4],
                    "spatial_dim": [32, 64], "dropout": [0.0, 0.1, 0.2]},
    "stgcn":       {"channels": [(32, 32, 64, 64), (64, 64, 128, 128)],
                    "Ks": [2, 3], "Kt": [2, 3], "dropout": [0.0, 0.1]},
    "gwnet":       {"residual_ch": [32, 64], "layers": [4, 6],
                    "dropout": [0.1, 0.3]},
    "dcrnn":       {"hidden_dim": [32, 64, 128], "max_diffusion_step": [1, 2, 3]},
    "astgcn":      {"channels": [32, 64], "Ks": [2, 3], "dropout": [0.0, 0.1]},
    "staeformer":  {"d_model": [32, 64, 128], "num_layers": [2, 3, 4],
                    "heads": [2, 4], "dropout": [0.0, 0.1, 0.2]},
}

def build_model(name, data, args):
    n_nodes = len(data["sensors"])
    in_dim = data["X_train"].shape[-1]
    T, theta = data["input_len"], data["pred_len"]
    adj = data["graphs"]["distance"]

    if name == "mgst":
        return MGSTTransformer(
            in_dim=in_dim, num_nodes=n_nodes, d_model=args.d_model,
            gnn_layers=args.gnn_layers, nhead=args.heads,
            num_layers=args.trans_layers, dim_feedforward=args.ff,
            dropout=args.dropout, pred_len=theta,
            num_graphs=len(args.graphs), fusion_mode=args.fusion,
            gate_dim=args.gate_dim, ctx_dim=args.ctx_dim,
            collect_alpha=args.dump_alpha), True

    if name == "lstm":
        return PerNodeTemporal(n_nodes, in_dim, theta, hidden=args.hidden,
                               num_layers=args.num_layers, dropout=args.dropout,
                               kind="lstm"), False
    if name == "gru":
        return PerNodeTemporal(n_nodes, in_dim, theta, hidden=args.hidden,
                               num_layers=args.num_layers, dropout=args.dropout,
                               kind="gru"), False
    if name == "tcn":
        return PerNodeTemporal(n_nodes, in_dim, theta, hidden=args.hidden,
                               num_layers=args.num_layers, dropout=args.dropout,
                               kind="tcn"), False
    if name == "transformer":
        return PerNodeTemporal(n_nodes, in_dim, theta, hidden=args.hidden,
                               num_layers=args.num_layers, dropout=args.dropout,
                               kind="transformer", nhead=args.heads, ff=args.ff), False
    if name == "stid":
        return STID(n_nodes, pred_len=theta, input_len=T, hidden=args.hidden,
                    num_layers=args.num_layers, spatial_dim=args.spatial_dim,
                    dropout=args.dropout), False
    if name == "stgcn":
        return STGCN(adj, pred_len=theta, input_len=T, Ks=args.Ks, Kt=args.Kt,
                     channels=tuple(args.channels), dropout=args.dropout), False
    if name == "gwnet":
        return GraphWaveNet(adj, pred_len=theta, input_len=T, layers=args.layers,
                            residual_ch=args.residual_ch,
                            dropout=args.dropout), False
    if name == "dcrnn":
        return DCRNN(adj, pred_len=theta, hidden_dim=args.hidden_dim,
                     max_diffusion_step=args.max_diffusion_step), False
    if name == "astgcn":
        return ASTGCN(adj, pred_len=theta, input_len=T, K=args.Ks,
                      channels=args.channels, dropout=args.dropout), False
    if name == "staeformer":
        return STAEformer(n_nodes, pred_len=theta, input_len=T,
                          d_model=args.d_model, nhead=args.heads,
                          num_layers=args.num_layers, ff=args.ff,
                          dropout=args.dropout), False
    raise ValueError(f"unknown model {name}")

def apply_feature_mask(x, mode):
    x = x.copy() if isinstance(x, np.ndarray) else x.clone()
    if mode == "no_speed":
        x[..., 1] = 0.0
    elif mode == "no_time":
        x[..., 2:] = 0.0
    return x

def _cfg_with(args, overrides):
    import copy
    cfg = copy.copy(args)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

def _train_trial(name, data, args, overrides, device, A_list, loaders,
                 epochs, seed):
    set_seed(seed, deterministic=False)
    cfg = _cfg_with(args, overrides)
    model, needs_graph = build_model(name, data, cfg)
    model = model.to(device)
    params = count_parameters(model)
    _A = A_list if needs_graph else None
    train_loader, val_loader, _ = loaders
    model, history = train_model(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=cfg.lr, weight_decay=cfg.weight_decay,
        patience=args.search_patience, A_list=_A, verbose=False,
        amp=not args.no_amp)
    val_metrics, _, _, _ = evaluate_model(model, val_loader, data["y_scaler"],
                                          device, A_list=_A)
    return val_metrics, params, history

def coordinate_search(args, data, device, A_list, loaders, csv_path=None):
    """Control-variable search. Returns (best_config, best_val_rmse, log).

    Two cheap safeguards, both invisible to the protocol:
      * a finished trial is never trained twice (each round probes the current
        best value again, and a resumed run would repeat earlier trials);
      * the log is written after every trial, so killing the job mid-search
        keeps what has already been measured.
    """
    space = SEARCH_SPACE.get(args.model)
    if not space:
        return {}, float("inf"), []

    current = {}
    for k, values in space.items():
        cur = getattr(args, k, None)
        current[k] = cur if cur in values else values[0]
    start_tag = (0, "(start)", str(current))     # before any replay below

    log = []
    cache = {}      # config -> (val metrics, #params), so nothing runs twice

    def cfg_key(cfg):
        return tuple(sorted((k, str(v)) for k, v in cfg.items()))

    def save():
        if csv_path:
            pd.DataFrame(log).to_csv(csv_path, index=False)

    recorded = {}
    if csv_path and os.path.exists(csv_path) and not args.force_search:
        old = pd.read_csv(csv_path)
        log.extend(old.to_dict("records"))
        for row in old.to_dict("records"):
            recorded[(int(row["round"]), str(row["param"]),
                      str(row["value"]))] = float(row["val_rmse"])
        for r in sorted(old["round"].unique()):
            if int(r) == 0:
                continue
            sub = old[old["round"] == r]
            for k, values in space.items():
                s = sub[sub["param"] == k]
                if s.empty:
                    continue
                best = str(s.loc[s["val_rmse"].idxmin(), "value"])
                current[k] = next((v for v in values if str(v) == best),
                                  current[k])
        if log:
            print(f"[search] resuming from {csv_path} "
                  f"({len(log)} trials already done)", flush=True)

    def run(cfg_over, row):
        k = cfg_key(cfg_over)
        if k not in cache:
            m, p, _ = _train_trial(args.model, data, args, cfg_over, device,
                                   A_list, loaders, args.search_epochs,
                                   args.search_seed)
            cache[k] = (m, p)
        m, p = cache[k]
        row.update({"val_rmse": m["RMSE"], "val_mae": m["MAE"], "params": p})
        log.append(row)
        save()
        return m

    best_rmse = float("inf")
    if start_tag in recorded:
        best_rmse = recorded[start_tag]
        row0 = old[old["round"] == 0].iloc[0]
        cache[cfg_key(current)] = (
            {"RMSE": best_rmse, "MAE": float(row0["val_mae"])},
            int(row0["params"]))
        print(f"[search] start  RMSE_val={best_rmse:.3f} (already done)", flush=True)
    else:
        m = run(current, {"round": 0, "param": "(start)", "value": str(current),
                          "selected": False})
        best_rmse = m["RMSE"]
        print(f"[search] start  RMSE_val={m['RMSE']:.3f}  cfg={current}",
              flush=True)

    for r in range(1, args.search_rounds + 1):
        for k, values in space.items():
            scored = []
            for v in values:
                tag = (r, k, str(v))
                if tag in recorded:
                    scored.append((recorded[tag], v))
                    continue
                trial = dict(current)
                trial[k] = v
                m = run(trial, {"round": r, "param": k, "value": str(v),
                                "selected": False})
                scored.append((m["RMSE"], v))
            rmse_v, best_v = min(scored, key=lambda t: t[0])
            current[k] = best_v
            print(f"[search] r{r} {k} -> {best_v} "
                  f"(RMSE_val={rmse_v:.3f})", flush=True)
            if rmse_v < best_rmse:
                best_rmse = rmse_v

    for row in log:
        if row["param"] in current and str(current[row["param"]]) == row["value"]:
            row["selected"] = True
    print(f"[search] BEST  RMSE_val={best_rmse:.3f}  cfg={current}", flush=True)
    return current, best_rmse, log

def _check_gpu(device):
    """Fail loudly (and early) when the installed PyTorch cannot run this GPU.

    Without this, a Blackwell card + an old PyTorch gives the cryptic
    "no kernel image is available for execution on the device" once per
    configuration, forty times in a row.
    """
    if device.type != "cuda":
        return
    cap = torch.cuda.get_device_capability(0)
    print(f"[device] {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}",
          flush=True)
    try:
        a = torch.zeros(2, 2, device=device)
        torch.matmul(a, a)
        torch.cuda.synchronize()
    except RuntimeError as e:
        raise SystemExit(
            f"[FATAL] 这块 GPU 跑不起来：{e}\n"
            "通常是 PyTorch 版本太旧，不支持该卡的 compute capability（5090 需要 "
            "sm_120，即 torch>=2.7）。重装：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
            "（cu128 也可以：.../whl/cu128）")

def run_one(args):
    device = get_device(args.gpu)
    print(f"[device] {device}", flush=True)
    _check_gpu(device)
    data = load_preprocessed(args.data_dir, args.tag,
                             input_len=args.input_len, pred_len=args.pred_len)

    A_list = None
    if (A_list is None) and args.model == "mgst":
        A_list = [torch.tensor(data["graphs"][g], device=device) for g in args.graphs]

    if args.model == "mgst" and args.mask != "all":
        for split in ("X_train", "X_val", "X_test"):
            data[split] = apply_feature_mask(data[split], args.mask)

    if args.inherit:
        inherit_json = os.path.join(args.out, f"{args.inherit}_best.json")
        if os.path.exists(inherit_json):
            with open(inherit_json, encoding="utf-8") as f:
                best_cfg = json.load(f)["best_config"]
            for k, v in best_cfg.items():
                setattr(args, k, tuple(v) if isinstance(v, list) else v)
            print(f"[inherit] {args.inherit}: {best_cfg}", flush=True)
        else:
            raise SystemExit(
                f"[FATAL] {inherit_json} 不存在，无法继承超参。\n"
                f"        先跑：--model mgst --fusion node --search "
                f"（它会写出 {inherit_json}）")

    if args.search and args.model != "naive":
        base = os.path.join(args.out, args.name)
        if os.path.exists(f"{base}_best.json") and not args.force_search:
            with open(f"{base}_best.json", encoding="utf-8") as f:
                saved = json.load(f)
            best_cfg = saved["best_config"]
            for k, v in best_cfg.items():
                setattr(args, k, tuple(v) if isinstance(v, list) else v)
            print(f"[search] reusing {base}_best.json: {best_cfg}", flush=True)
            if args.search_only:
                return None
        else:
            set_seed(args.search_seed)
            search_loaders = create_dataloaders(
                data["X_train"], data["X_val"], data["X_test"],
                data["y_train"], data["y_val"], data["y_test"],
                batch_size=args.batch_size, num_workers=args.num_workers,
                seed=args.search_seed, input_len=args.input_len,
                pred_len=args.pred_len)
            best_cfg, best_rmse, log = coordinate_search(
                args, data, device, A_list, search_loaders,
                csv_path=f"{base}_search.csv")
            with open(f"{base}_best.json", "w", encoding="utf-8") as f:
                json.dump({"model": args.model, "tag": data.get("tag"),
                           "criterion": "val RMSE", "protocol":
                           "control-variable "
                           f"(coordinate descent, {args.search_rounds} rounds, "
                           f"{args.search_epochs} epochs/trial, seed "
                           f"{args.search_seed})",
                           "best_val_rmse": best_rmse, "best_config":
                           {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in best_cfg.items()}}, f, indent=2)
            print(f"[search] saved {base}_search.csv and {base}_best.json",
                  flush=True)
            for k, v in best_cfg.items():
                setattr(args, k, v)
            if args.search_only:
                return pd.DataFrame(log)
            del search_loaders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []
    for seed in args.seeds:
        name = f"{args.name}_seed{seed}"
        done_csv = os.path.join(args.out, f"{name}_overall.csv")
        alpha_npz = os.path.join(args.out, f"{name}_alpha.npz")
        want_alpha = args.dump_alpha and args.model == "mgst"
        if (os.path.exists(done_csv) and not args.force
                and not (want_alpha and not os.path.exists(alpha_npz))):
            print(f"[skip] {name} already in {args.out}", flush=True)
            rows.append(pd.read_csv(done_csv).iloc[0].to_dict())
            continue

        set_seed(seed)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        t0 = time.time()

        if args.model == "naive":
            overall, per_horizon, preds, trues = naive_metrics(data, data["y_scaler"])
            params, history = 0, {"train": [], "val": [], "best_val": np.nan,
                                  "best_epoch": 0}
        else:
            loaders = create_dataloaders(
                data["X_train"], data["X_val"], data["X_test"],
                data["y_train"], data["y_val"], data["y_test"],
                batch_size=args.batch_size, num_workers=args.num_workers,
                seed=seed, input_len=args.input_len, pred_len=args.pred_len)
            train_loader, val_loader, test_loader = loaders

            model, needs_graph = build_model(args.model, data, args)
            model = model.to(device)
            params = count_parameters(model)

            _A = A_list if needs_graph else None
            if args.compile:
                try:
                    model = torch.compile(model)
                    print("[compile] torch.compile enabled", flush=True)
                except Exception as e:
                    print(f"[compile] skipped ({e})", flush=True)

            model, history = train_model(
                model, train_loader, val_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                patience=args.patience, A_list=_A, verbose=args.verbose,
                amp=not args.no_amp)
            overall, per_horizon, preds, trues = evaluate_model(
                model, test_loader, data["y_scaler"], device, A_list=_A)

            if args.dump_alpha and args.model == "mgst":
                alpha, hours = model.collect_fusion_weights(test_loader, device, _A)
                if alpha is not None:
                    np.savez_compressed(
                        os.path.join(args.out, f"{args.name}_seed{seed}_alpha.npz"),
                        alpha=alpha, hour=hours)

        name = f"{args.name}_seed{seed}"
        cfg = dict(model=args.model, seed=seed, fusion=args.fusion,
                   graphs=",".join(args.graphs) if args.graphs else "",
                   mask=args.mask, params=params,
                   minutes=round((time.time() - t0) / 60, 1),
                   dataset_tag=data.get("tag", args.tag), data_dir=args.data_dir,
                   epochs=args.epochs, lr=args.lr, patience=args.patience,
                   batch_size=args.batch_size, search_space=SEARCH_SPACE.get(args.model))
        row = save_run(args.out, name, cfg, overall, per_horizon, history,
                       preds=preds if args.save_preds else None,
                       trues=trues if args.save_preds else None)
        row["params"] = params
        row["minutes"] = cfg["minutes"]
        row["best_epoch"] = history.get("best_epoch", 0)
        rows.append(row)
        print(f"[{name}] RMSE={overall['RMSE']:.3f} MAE={overall['MAE']:.3f} "
              f"R2={overall['R2']:.4f} ({row['minutes']} min)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.name}_all_seeds.csv"), index=False)
    if len(df) > 1:
        agg = df[["RMSE", "MAE", "sMAPE(%)", "R2", "EVS", "WAPE(%)"]].agg(["mean", "std"])
        agg.to_csv(os.path.join(args.out, f"{args.name}_summary.csv"))
        print("\n=== mean +- std over seeds ===")
        print(agg.round(4).to_string())
    return df

DEFAULT_DATA_DIR = os.environ.get("MGST_DATA_DIR",
                                  "/root/autodl-fs/preprocess_result")

def parse_args(argv=None):
    p = argparse.ArgumentParser("MGST-Transformer experiments")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--tag", default=None,
                   help="filename suffix: '' or '18年' for 2018, '24年' for 2024. "
                        "Omit it to auto-detect from the directory contents "
                        "(safest: the run prints which suffix it resolved to).")
    p.add_argument("--out", default="results")
    p.add_argument("--name", default=None)
    p.add_argument("--model", default="mgst",
                   choices=["mgst", "naive", "lstm", "gru", "tcn", "transformer",
                            "stid", "stgcn", "gwnet", "dcrnn", "astgcn",
                            "staeformer"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])

    p.add_argument("--fusion", default="node", choices=["global", "sample", "node"])
    p.add_argument("--graphs", nargs="+", default=ALL_GRAPHS)
    p.add_argument("--mask", default="all", choices=["all", "no_speed", "no_time"])
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--trans-layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--gate-dim", type=int, default=32)
    p.add_argument("--ctx-dim", type=int, default=16)
    p.add_argument("--dump-alpha", action="store_true")

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--spatial-dim", type=int, default=64)
    p.add_argument("--residual-ch", type=int, default=32)
    p.add_argument("--channels", type=int, nargs=4, default=[32, 32, 64, 64])
    p.add_argument("--Ks", type=int, default=3)
    p.add_argument("--Kt", type=int, default=3)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--max-diffusion-step", type=int, default=2)

    p.add_argument("--input-len", type=int, default=12)
    p.add_argument("--pred-len", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 4))
    p.add_argument("--gpu", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print-protocol", action="store_true")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (on by default on CUDA)")
    p.add_argument("--force", action="store_true",
                   help="re-train even if the output files already exist")
    p.add_argument("--force-search", action="store_true",
                   help="re-run the search even if <name>_best.json exists")

    p.add_argument("--search", action="store_true",
                   help="search SEARCH_SPACE on the validation split, then run "
                        "the final multi-seed evaluation with the best config")
    p.add_argument("--search-only", action="store_true",
                   help="run the search and stop (no final multi-seed run)")
    p.add_argument("--search-epochs", type=int, default=30,
                   help="training budget per trial (vs --epochs for the final run)")
    p.add_argument("--search-rounds", type=int, default=2,
                   help="coordinate-descent sweeps over all factors")
    p.add_argument("--search-seed", type=int, default=42)
    p.add_argument("--search-patience", type=int, default=5,
                   help="early-stopping patience during a search trial "
                        "(the final runs keep --patience)")
    p.add_argument("--inherit", default=None,
                   help="load the tuned hyper-parameters of <out>/<inherit>_best.json "
                        "instead of the argparse defaults, so that an ablation "
                        "differs from the full model in ONE factor only")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (opt-in: first epoch is slow)")
    p.add_argument("--save-preds", action="store_true",
                   help="save test predictions (<name>_seed<n>_pred.npz, fp16). "
                        "Only needed for the prediction-plot figures; off by "
                        "default because the arrays are large")
    args, unknown = p.parse_known_args(argv)
    print(f"[build] run_experiments.py {BUILD}  (search supported: True)",
          flush=True)
    if unknown:
        print(f"[warn] ignoring unrecognised arguments: {unknown}", flush=True)
        if "--search" in unknown:
            print("[FATAL] this copy of run_experiments.py predates the "
                  "hyper-parameter search. Re-copy the file from the local "
                  "machine; results without --search use default hyper-parameters.",
                  flush=True)
    return args

def main(argv=None):
    args = parse_args(argv)
    if args.print_protocol:
        print(json.dumps(SEARCH_SPACE, indent=2))
        return
    os.makedirs(args.out, exist_ok=True)
    if args.name is None:
        suffix = "" if len(args.graphs) == 4 else "_" + "".join(g[0] for g in args.graphs)
        args.name = f"{args.model}_{args.fusion}{suffix}" if args.model == "mgst" \
            else args.model
    run_one(args)

if __name__ == "__main__":
    main()
