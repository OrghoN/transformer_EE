# ========================= train.py =========================

"""A module for training a model – revised with richer bookkeeping."""

import json
import os
import math
from typing import Dict, List

import numpy as np
import torch
import pandas as pd

from transformer_ee.dataloader.load import get_train_valid_test_dataloader
from transformer_ee.logger.train_logger import BaseLogger
from transformer_ee.model import create_model
from transformer_ee.utils import get_gpu, hash_dict

from .loss import (
    linear_combination_loss,
    abs_invariant_mass_squared_prediction_loss,
    loss_function,  # expose base losses for per‑component tracking
)
from .loss_track import plot_loss
from .energy_plots import DEFAULT_RECO_ESTIMATORS, default_bin_edges, plot_energy_dependence
from .optimizer import create_optimizer

# -----------------------------------------------------------------------------
# Alpha / Beta schedules (unchanged)
# -----------------------------------------------------------------------------

def alpha(epoch: float, a_start: float = 1.0, a_end: float = 1.0, k: float = 1.0, x0: float = 6.0) -> float:
    """Decreasing S‑curve from a_start→a_end over ~10 epochs by default."""
    return a_end + (a_start - a_end) / (1 + math.exp(k * (epoch - x0)))


def beta(epoch: float, b_start: float = 0.01, b_end: float = 1.0, k: float = 1.0, x0: float = 6.0) -> float:
    """Increasing S‑curve from b_start→b_end over ~10 epochs by default."""
    return b_start + (b_end - b_start) / (1 + math.exp(-k * (epoch - x0)))

# -----------------------------------------------------------------------------
# Memory monitoring
# -----------------------------------------------------------------------------

def _memory_in_use_gib() -> float | None:
    """
    Memory used by this process and all its descendants (DataLoader workers), in GiB, as the
    sum of PSS from /proc/<pid>/smaps_rollup: pages shared copy-on-write between the trainer
    and its forked workers are counted once, as a memory limit counts them.
    Returns None where /proc does not provide this (non-Linux).
    """
    try:
        children: Dict[int, List[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", encoding="ascii") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))
        todo, total_kib = [os.getpid()], 0
        while todo:
            pid = todo.pop()
            todo.extend(children.get(pid, []))
            try:
                with open(f"/proc/{pid}/smaps_rollup", encoding="ascii") as f:
                    for line in f:
                        if line.startswith("Pss:"):
                            total_kib += int(line.split()[1])
                            break
            except OSError:
                continue
        return total_kib / 2**20
    except OSError:
        return None


# -----------------------------------------------------------------------------
# Trainer class
# -----------------------------------------------------------------------------

class MVtrainer:
    r"""
    Multivariate trainer with detailed loss breakdown.
    """

    # ------------------------------------------------------------------
    # construction & bookkeeping containers
    # ------------------------------------------------------------------

    def __init__(self, input_d: dict, logger: BaseLogger | None = None):
        self.gpu_device = get_gpu()
        self.input_d = input_d
        self.logger = logger
        print(json.dumps(self.input_d, indent=4))
        # A single, ever-increasing step counter for W&B
        self.global_step = 0

        # dataloaders & model
        (
            self.trainloader,
            self.validloader,
            self.testloader,
            self.train_set_stat,
        ) = get_train_valid_test_dataloader(input_d)
        self.net = create_model(self.input_d).to(self.gpu_device)
        self.optimizer = create_optimizer(self.input_d, self.net)
        # Tracks the best (minimum) validation loss observed so far.
        # For minimization objectives, +inf is the correct scale-independent sentinel.
        self.best_val_loss_sentinel = float("inf")
        self.best_model_epoch: int | None = None
        self.best_model_val_loss: float | None = None
        self.last_model_epoch: int | None = None
        self.last_model_val_loss: float | None = None
        self.print_interval = self.input_d.get("print_interval", 1)

        # Checkpoint retention policy.
        # keep_last_n_epoch_checkpoints controls optional rolling files
        # like last_model_epoch_0010.zip used for long runs and diagnostics.
        ckpt_cfg = self.input_d.get("checkpointing", {})
        default_keep_last_n = (
            int(self.input_d.get("early_stopping", {}).get("window", 5))
            if self.input_d.get("early_stopping", {}).get("enabled", False)
            else 0
        )
        self.keep_last_n_epoch_checkpoints = max(
            0, int(ckpt_cfg.get("keep_last_n_epoch_checkpoints", default_keep_last_n))
        )
        self._epoch_checkpoint_history: List[str] = []

        # Optional early stopping.
        # Stops when percent improvement in val loss over a recent window
        # drops below min_delta_pct.
        early_cfg = self.input_d.get("early_stopping", {})
        self.early_stopping_enabled = bool(early_cfg.get("enabled", False))
        configured_window = int(early_cfg.get("window", 5))
        if self.early_stopping_enabled and configured_window < 2:
            raise ValueError(
                "early_stopping.window must be >= 2 when early stopping is enabled. "
                f"Received {configured_window}."
            )
        self.early_stopping_window = (
            max(2, configured_window)
            if self.early_stopping_enabled
            else max(1, configured_window)
        )

        default_min_delta_pct = 0.1
        self.early_stopping_min_delta_pct = float(
            early_cfg.get("min_delta_pct", default_min_delta_pct)
        )
        if self.early_stopping_enabled and self.early_stopping_min_delta_pct <= 0.0:
            raise ValueError(
                "early_stopping.min_delta_pct must be > 0 when early stopping is enabled. "
                f"Received {self.early_stopping_min_delta_pct}."
            )
        self.early_stop_triggered = False
        self.early_stop_epoch: int | None = None

        # epoch‑wise aggregates (legacy)
        self.train_loss_list_per_epoch: List[float] = []
        self.valid_loss_list_per_epoch: List[float] = []

        # NEW – component‑wise LCL contributions (shape = [epochs, n_vars])
        self.train_comp_loss_per_epoch: List[torch.Tensor] = []
        self.valid_comp_loss_per_epoch: List[torch.Tensor] = []

        # NEW – specialised trackers for PredInvMass scheme
        self.alpha_list: List[float] = []
        self.beta_list: List[float] = []
        self.train_pred_mass_per_epoch: List[float] = []
        self.valid_pred_mass_per_epoch: List[float] = []
        self.train_alpha_LCL_per_epoch: List[float] = []
        self.train_beta_PM_per_epoch: List[float] = []
        self.valid_alpha_LCL_per_epoch: List[float] = []
        self.valid_beta_PM_per_epoch: List[float] = []

        # misc
        self.epochs = input_d["model"].get("epochs", 100)
        print("epochs: ", self.epochs)
        self.save_path = os.path.join(input_d.get("save_path", "."), "model_" + hash_dict(self.input_d))
        os.makedirs(self.save_path, exist_ok=True)
        self._dump_static_json()

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------

    def _dump_static_json(self):
        input_json = os.path.join(self.save_path, "input.json")
        if not os.path.exists(input_json):
            with open(input_json, "w") as f:
                json.dump(self.input_d, f, indent=4)

        stat_json = os.path.join(self.save_path, "trainset_stat.json")
        if not os.path.exists(stat_json):
            with open(stat_json, "w") as f:
                json.dump(self.train_set_stat, f, indent=4)

    # ------------------------------------------------------------------
    # core training loops – LCL only
    # ------------------------------------------------------------------

    def _check_tensor(self, name: str, t: torch.Tensor, max_print: int = 3):
        """Debug helper: print NaN/Inf and basic stats for a tensor."""
        if t is None:
            print(f"[CHECK] {name}: is None")
            return

        if not isinstance(t, torch.Tensor):
            print(f"[CHECK] {name}: not a tensor (type={type(t)})")
            return

        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        print(
            f"[CHECK] {name}: shape={tuple(t.shape)}, "
            f"nan={has_nan}, inf={has_inf}, "
            f"min={t.min().item() if t.numel() > 0 else 'NA'}, "
            f"max={t.max().item() if t.numel() > 0 else 'NA'}"
        )
        if has_nan or has_inf:
            flat = t.flatten()
            idx = torch.nonzero(torch.isnan(flat) | torch.isinf(flat)).squeeze()
            idx = idx[:max_print] if idx.numel() > max_print else idx
            print(f"  -> first problematic indices in {name}: {idx.tolist()}")
            print(f"  -> values at those indices: {flat[idx].tolist()}")

    def train_LCL(self):
        """Standard LCL training with per‑component loss tracking."""
        # torch.autograd.set_detect_anomaly(True) #debugging-----
        
        loss_kwargs = self.input_d["loss"]["kwargs"]
        base_loss_names = loss_kwargs["base_loss_names"]
        coeffs = loss_kwargs["coefficients"]
        n_vars = len(base_loss_names)

        # Monitoring: loss.png, accuracy_<target>.png and energy_dependence_<target>.png are
        # redrawn once per epoch, after the end-of-epoch validation pass.

        for epoch in range(self.epochs):
            # ------------------- training -------------------
            self.net.train()
            epoch_total_loss = []
            comp_sum = torch.zeros(n_vars, device=self.gpu_device)

            for batch_idx, batch in enumerate(self.trainloader):
                vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]

                # ---- ONE-TIME DEEP CHECK ON FIRST BATCH ----
                if epoch == 0 and batch_idx == 0:
                    print("\n=== DEBUG: First training batch sanity check ===")
                    self._check_tensor("vec", vec)
                    self._check_tensor("sca", sca)
                    self._check_tensor("msk", msk)
                    self._check_tensor("tgt", tgt)
                    self._check_tensor("wgt", wgt)
                    # inspect per-target column ranges
                    for j in range(tgt.shape[1]):
                        col = tgt[:, j]
                        self._check_tensor(f"tgt[:, {j}]", col)
                    print("=== END first batch sanity check ===\n")

                out = self.net.forward(vec, sca, msk)

                if epoch == 0 and batch_idx == 0:
                    print("\n=== DEBUG: First forward pass output ===")
                    self._check_tensor("out", out)
                    for j in range(out.shape[1]):
                        self._check_tensor(f"out[:, {j}]", out[:, j])
                    print("=== END forward output check ===\n")

                # total loss (legacy)
                lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                if epoch == 0 and batch_idx == 0:
                    print(f"[DEBUG] lcl_loss on first batch: {lcl_loss}")

                # component‑wise contributions (un‑weighted by alpha/beta)
                for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                    comp_j = (
                        coef
                        * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                        .detach()
                    )
                    comp_sum[j] += comp_j

                # step
                self.optimizer.zero_grad()
                lcl_loss.backward()

                if epoch == 0 and batch_idx == 0:
                    print("\n=== DEBUG: Gradient check (first batch) ===")
                    bad_params = []
                    for name, p in self.net.named_parameters():
                        if p.grad is None:
                            continue
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            bad_params.append(name)
                    if bad_params:
                        print("NaN/Inf gradients in parameters:", bad_params)
                    else:
                        print("No NaN/Inf gradients detected in first batch.")
                    print("=== END gradient check ===\n")
        
                self.optimizer.step()

                epoch_total_loss.append(lcl_loss.detach())
                # ---- W&B per-step logging (main process only) ----
                if self.logger is not None:
                    # ensure plain Python floats; also log a monotonic step
                    self.logger.log({
                        "train/loss": float(lcl_loss.detach().item()),
                        "train/step": int(self.global_step),
                        "epoch": int(epoch),
                    }, step=int(self.global_step))
                self.global_step += 1


                if batch_idx % self.print_interval == 0:
                    print(f"Epoch: {epoch}, batch: {batch_idx} Loss: {lcl_loss:0.4f}")


            # epoch aggregates
            self.train_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_total_loss)))
            self.train_comp_loss_per_epoch.append(comp_sum / len(self.trainloader))

            # ------------------- validation -------------------
            val_loss, val_comp, val_out, val_tgt = self._validate_LCL(
                loss_kwargs, base_loss_names, coeffs, n_vars
            )
            self.valid_loss_list_per_epoch.append(val_loss)
            self.valid_comp_loss_per_epoch.append(val_comp)

            # end-of-epoch monitoring point (plots + CSVs)
            self._record_epoch(
                epoch + 1, epoch_total_loss, self.train_comp_loss_per_epoch[-1],
                val_loss, val_out, val_tgt,
            )

            # ---- Optional: epoch-level summaries (once per epoch) ----
            if self.logger is not None:
                tr = float(self.train_loss_list_per_epoch[-1].detach().cpu().item()
                           if hasattr(self.train_loss_list_per_epoch[-1], "detach") else
                           float(self.train_loss_list_per_epoch[-1]))
                va = float(self.valid_loss_list_per_epoch[-1].detach().cpu().item()
                           if hasattr(self.valid_loss_list_per_epoch[-1], "detach") else
                           float(self.valid_loss_list_per_epoch[-1]))
                # batch components (example: log as a small dict)
                comp = self.train_comp_loss_per_epoch[-1].detach().cpu().tolist()
                comp_dict = {f"train/comp_{j}": float(x) for j, x in enumerate(comp)}
                self.logger.log({
                    "epoch": int(epoch),
                    "train/loss_epoch": tr,
                    "valid/loss_epoch": va,
                    **comp_dict,
                    "train/step": int(self.global_step),  # keeps charts aligned
                }, step=int(self.global_step))

            # ------------------- bookkeeping & plots -------------------
            # loss.png is drawn by _record_epoch, not here.
            should_stop = self._checkpoint_and_plot(epoch, scheme="LCL", n_vars=n_vars, plot=False)
            if should_stop:
                break

        self._finalize_checkpoint_layout()

        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------
    # per-epoch monitoring helpers (used by train_LCL)
    # ------------------------------------------------------------------

    def _validate_LCL(self, loss_kwargs, base_loss_names, coeffs, n_vars):
        """One pass over the validation set.

        Returns (mean LCL loss, per-component loss, predictions, targets); predictions and
        targets are CPU numpy arrays of shape (n_valid, n_targets) in physical units (the
        targets are not normalised by the dataset).
        """
        self.net.eval()
        val_losses = []
        comp_val_sum = torch.zeros(n_vars, device=self.gpu_device)
        outs, tgts = [], []
        with torch.no_grad():
            for batch in self.validloader:
                vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                out = self.net.forward(vec, sca, msk)
                lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                    comp_val_sum[j] += (
                        coef * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt)).detach()
                    )
                val_losses.append(lcl_loss.detach())
                outs.append(out.detach().cpu().numpy())
                tgts.append(tgt.detach().cpu().numpy())
        return (
            torch.mean(torch.tensor(val_losses)),
            comp_val_sum / len(self.validloader),
            np.concatenate(outs),
            np.concatenate(tgts),
        )

    @staticmethod
    def _accuracy_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """Accuracy of one target on the validation set, from r = (pred - true) / true.

        Only events with true > 0 are used, so the trueLepE = -0.5 placeholder of background
        events is excluded (same convention as the masked MAPE loss). Unweighted.
        """
        sel = true > 0
        if not np.any(sel):
            nan = float("nan")
            return {"mean": nan, "median": nan, "std": nan, "q68": nan, "n": 0}
        r = (pred[sel] - true[sel]) / true[sel]
        q16, q50, q84 = np.percentile(r, [16, 50, 84])
        return {
            "mean": float(np.mean(r)),
            "median": float(q50),
            "std": float(np.std(r)),
            "q68": float(0.5 * (q84 - q16)),
            "n": int(sel.sum()),
        }

    def _record_epoch(self, epoch_pos, train_losses, train_comp, val_loss, val_out, val_tgt):
        """Store one per-epoch point (epoch_pos = number of completed epochs) and redraw loss.png (linear y-axis),
        energy_dependence_<target>.png, accuracy_<target>.png and training_history.csv in
        save_path."""
        if not hasattr(self, "epoch_history"):
            self.epoch_history: List[Dict[str, float]] = []

        target_names = self.input_d["target"]
        row: Dict[str, float] = {
            "epoch": float(epoch_pos),
            "train_loss": float(torch.mean(torch.tensor(train_losses)).item()) if train_losses else float("nan"),
            "valid_loss": float(val_loss.item()),
        }
        for j, name in enumerate(target_names):
            row[f"train_loss_{name}"] = float(train_comp[j].item())
        for j, name in enumerate(target_names):
            for k, v in self._accuracy_metrics(val_out[:, j], val_tgt[:, j]).items():
                row[f"{name}_{k}"] = v
        self.epoch_history.append(row)

        hist = self.epoch_history
        x = [h["epoch"] for h in hist]

        # loss.png, one point per epoch
        metrics: Dict[str, List] = {
            "Training Loss Total": [h["train_loss"] for h in hist],
            "Validation Loss Total": [h["valid_loss"] for h in hist],
        }
        for j, name in enumerate(target_names):
            metrics[f"Train: Var. {j} ({name})"] = [h[f"train_loss_{name}"] for h in hist]
        plot_loss(metrics, self.save_path, x=x, logy=False)

        # energy_dependence_<target>.png: accuracy vs true energy, model vs standard reco
        self._plot_energy_dependence(epoch_pos, val_out, val_tgt)

        # accuracy_<target>.png, one per target
        for name in target_names:
            acc = {
                "Bias: mean of (pred-true)/true": [h[f"{name}_mean"] for h in hist],
                "Bias: median of (pred-true)/true": [h[f"{name}_median"] for h in hist],
                "Resolution: RMS of (pred-true)/true": [h[f"{name}_std"] for h in hist],
                "Resolution: 68% half-width of (pred-true)/true": [h[f"{name}_q68"] for h in hist],
            }
            safe = name.replace("/", "_").replace(" ", "_")
            plot_loss(
                acc, self.save_path, x=x, filename=f"accuracy_{safe}.png", logy=False,
                ylabel="(pred - true) / true", title=f"Validation accuracy: {name}", zero_line=True,
            )

        pd.DataFrame(hist).to_csv(os.path.join(self.save_path, "training_history.csv"), index=False)

        summary = ", ".join(
            f"{name}: bias {row[f'{name}_median']:+.4f}, res68 {row[f'{name}_q68']:.4f}"
            for name in target_names
        )
        mem = _memory_in_use_gib()
        mem_txt = f" | host memory {mem:.2f} GiB" if mem is not None else ""
        print(
            f"[epoch {epoch_pos:.0f} done] train_loss: {row['train_loss']:0.4f}, "
            f"valid_loss: {row['valid_loss']:0.4f} | {summary}{mem_txt}"
        )

    def _valid_eval_columns(self) -> Dict[str, np.ndarray]:
        """Raw config "eval_columns" of the validation set (row order = validloader order,
        which does not shuffle), as name -> float array. Read once and cached."""
        if not hasattr(self, "_valid_extra_cache"):
            cols: Dict[str, np.ndarray] = {}
            extra = getattr(self.validloader.dataset, "extra", None)
            if extra is not None and getattr(extra, "width", 0) > 0:
                for name in extra.columns:
                    cols[name] = extra[name].to_numpy().astype(np.float64)
            self._valid_extra_cache = cols
        return self._valid_extra_cache

    def _plot_energy_dependence(self, epoch_pos, val_out, val_tgt):
        """Per target, binned in true energy: median estimate, bias and resolution of the model
        and of the standard reco estimators, on the validation set.

        Writes save_path/energy_dependence_<target>.png (latest), a snapshot per epoch in
        save_path/energy_dependence/, and writes the binned numbers of all epochs so far
        to energy_dependence.csv.
        Config (optional): "reco_comparison" {target: [eval column, ...]} and
        "energy_bins" {target: [bin edges in GeV]}.
        """
        extra = self._valid_eval_columns()
        reco_cfg = self.input_d.get("reco_comparison", DEFAULT_RECO_ESTIMATORS)
        bins_cfg = self.input_d.get("energy_bins", {})
        if not hasattr(self, "_energy_bin_edges"):
            self._energy_bin_edges: Dict[str, np.ndarray] = {}

        n_valid = len(val_tgt)
        selections = {"all validation events": np.ones(n_valid, dtype=bool)}
        if "passAnaFull" in extra and len(extra["passAnaFull"]) == n_valid:
            selections["passAnaFull (analysis selection)"] = extra["passAnaFull"] > 0.5

        rows = []
        for j, name in enumerate(self.input_d["target"]):
            true = val_tgt[:, j].astype(np.float64)
            estimators = {"transformer": val_out[:, j].astype(np.float64)}
            for col in reco_cfg.get(name, []):
                if col in extra and len(extra[col]) == n_valid:
                    estimators[col] = extra[col]
                elif not getattr(self, "_warned_missing_reco", False):
                    print(f"[WARN] reco estimator '{col}' for {name} is not in eval_columns; skipped")
            # bins are fixed at the first call so that plots of different epochs are comparable
            if name not in self._energy_bin_edges:
                self._energy_bin_edges[name] = (
                    np.asarray(bins_cfg[name], dtype=float) if name in bins_cfg
                    else default_bin_edges(true)
                )
            safe = name.replace("/", "_").replace(" ", "_")
            snapshot = os.path.join(
                self.save_path, "energy_dependence", f"energy_dependence_{safe}_epoch{int(round(epoch_pos)):03d}.png"
            )
            latest = os.path.join(self.save_path, f"energy_dependence_{safe}.png")
            rows += plot_energy_dependence(
                true, estimators, selections, self._energy_bin_edges[name], [snapshot, latest],
                target=name, epoch=epoch_pos,
            )
        self._warned_missing_reco = True

        # kept in memory and rewritten whole, like training_history.csv
        if not hasattr(self, "_energy_dependence_rows"):
            self._energy_dependence_rows: List[Dict] = []
        self._energy_dependence_rows += rows
        pd.DataFrame(self._energy_dependence_rows).to_csv(
            os.path.join(self.save_path, "energy_dependence.csv"), index=False
        )

    # ------------------------------------------------------------------
    # core training loops – LCL + PredInvMass
    # ------------------------------------------------------------------

    def train_LCL_wPredInvMass(self):
        """Hybrid loss training with richer analytics."""
        loss_kwargs = self.input_d["loss"]["kwargs"]
        base_loss_names = loss_kwargs["base_loss_names"]
        coeffs = loss_kwargs["coefficients"]
        n_vars = len(base_loss_names)

        for epoch in range(self.epochs):
            a_val = alpha(epoch)
            b_val = beta(epoch)
            self.alpha_list.append(a_val)
            self.beta_list.append(b_val)

            # ------------------- training -------------------
            self.net.train()
            epoch_total_loss = []
            comp_sum = torch.zeros(n_vars, device=self.gpu_device)
            pm_sum = 0.0
            aLCL_sum = 0.0
            bPM_sum = 0.0

            for batch_idx, batch in enumerate(self.trainloader):
                vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                out = self.net.forward(vec, sca, msk)

                lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                pm_loss = abs_invariant_mass_squared_prediction_loss(out, weight=wgt)
                total_loss = a_val * lcl_loss + b_val * pm_loss

                # per component contribution of LCL (for diagnostics)
                for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                    comp_sum[j] += (
                        coef
                        * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                        .detach()
                    )

                # step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss.append(total_loss.detach())
                pm_sum += pm_loss.detach()
                aLCL_sum += (a_val * lcl_loss).detach()
                bPM_sum += (b_val * pm_loss).detach()

                if batch_idx % self.print_interval == 0:
                    print(f"Epoch: {epoch}, batch: {batch_idx} Loss: {total_loss:0.4f}")

            # epoch aggregates (train)
            n_train_batches = len(self.trainloader)
            self.train_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_total_loss)))
            self.train_comp_loss_per_epoch.append(comp_sum / n_train_batches)
            self.train_pred_mass_per_epoch.append(pm_sum / n_train_batches)
            self.train_alpha_LCL_per_epoch.append(aLCL_sum / n_train_batches)
            self.train_beta_PM_per_epoch.append(bPM_sum / n_train_batches)

            # ------------------- validation -------------------
            self.net.eval()
            epoch_val_loss = []
            comp_val_sum = torch.zeros(n_vars, device=self.gpu_device)
            pm_val_sum = 0.0
            aLCL_val_sum = 0.0
            bPM_val_sum = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validloader):
                    vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                    out = self.net.forward(vec, sca, msk)

                    lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                    pm_loss = abs_invariant_mass_squared_prediction_loss(out, weight=wgt)
                    total_loss = a_val * lcl_loss + b_val * pm_loss

                    for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                        comp_val_sum[j] += (
                            coef
                            * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                            .detach()
                        )

                    epoch_val_loss.append(total_loss.detach())
                    pm_val_sum += pm_loss.detach()
                    aLCL_val_sum += (a_val * lcl_loss).detach()
                    bPM_val_sum += (b_val * pm_loss).detach()

            n_val_batches = len(self.validloader)
            self.valid_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_val_loss)))
            self.valid_comp_loss_per_epoch.append(comp_val_sum / n_val_batches)
            self.valid_pred_mass_per_epoch.append(pm_val_sum / n_val_batches)
            self.valid_alpha_LCL_per_epoch.append(aLCL_val_sum / n_val_batches)
            self.valid_beta_PM_per_epoch.append(bPM_val_sum / n_val_batches)

            # ------------------- checkpoint & plots -------------------
            should_stop = self._checkpoint_and_plot(epoch, scheme="Hybrid", n_vars=n_vars)
            if should_stop:
                break

        self._finalize_checkpoint_layout()

        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------
    # helper – model checkpoint + plotting
    # ------------------------------------------------------------------

    def _write_checkpoint_metadata(self):
        """Persist compact checkpoint/training status metadata for reproducibility."""
        best_filename = "best_model.zip" if os.path.exists(os.path.join(self.save_path, "best_model.zip")) else None
        last_filename = "last_model.zip" if os.path.exists(os.path.join(self.save_path, "last_model.zip")) else None

        latest_filename = last_filename or best_filename
        latest_epoch = self.last_model_epoch if last_filename is not None else self.best_model_epoch
        latest_val_loss = (
            self.last_model_val_loss
            if last_filename is not None
            else self.best_model_val_loss
        )

        metadata = {
            "best_model": {
                "epoch": self.best_model_epoch,
                "validation_loss": self.best_model_val_loss,
                "filename": best_filename,
            },
            "last_model": {
                "epoch": self.last_model_epoch,
                "validation_loss": self.last_model_val_loss,
                "filename": last_filename,
            },
            "latest_checkpoint": {
                "epoch": latest_epoch,
                "validation_loss": latest_val_loss,
                "filename": latest_filename,
            },
            "keep_last_n_epoch_checkpoints": self.keep_last_n_epoch_checkpoints,
            "retained_last_model_epoch_checkpoints": self._epoch_checkpoint_history,
            "early_stopping": {
                "enabled": self.early_stopping_enabled,
                "window": self.early_stopping_window,
                "min_delta_pct": self.early_stopping_min_delta_pct,
                "triggered": self.early_stop_triggered,
                "triggered_epoch": self.early_stop_epoch,
            },
        }
        with open(os.path.join(self.save_path, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def _finalize_checkpoint_layout(self):
        """Keep best_model.zip canonical; keep last_model.zip only when it differs."""
        if self.best_model_epoch is not None and self.last_model_epoch == self.best_model_epoch:
            last_model_path = os.path.join(self.save_path, "last_model.zip")
            if os.path.exists(last_model_path):
                os.remove(last_model_path)
        self._write_checkpoint_metadata()

    def _should_early_stop(self) -> bool:
        """
        Trigger early stopping when validation loss improvement over the recent
        window is smaller than the configured percentage.
        """
        if not self.early_stopping_enabled:
            return False
        if len(self.valid_loss_list_per_epoch) < self.early_stopping_window:
            return False

        recent = [float(x.detach().cpu().item()) for x in self.valid_loss_list_per_epoch[-self.early_stopping_window :]]
        first = recent[0]
        best = min(recent)
        baseline = max(abs(first), 1e-12)
        improvement_pct = ((first - best) / baseline) * 100.0
        return improvement_pct < self.early_stopping_min_delta_pct

    def _checkpoint_and_plot(self, epoch: int, scheme: str, n_vars: int, plot: bool = True) -> bool:
        # stdout
        print(
            f"Epoch: {epoch}, train_loss: {self.train_loss_list_per_epoch[-1]:0.4f}, "
            f"valid_loss: {self.valid_loss_list_per_epoch[-1]:0.4f}"
        )

        current_val_loss = float(self.valid_loss_list_per_epoch[-1].detach().cpu().item())

        # Always write the latest model checkpoint for recovery/resume-style workflows.
        torch.save(self.net.state_dict(), os.path.join(self.save_path, "last_model.zip"))
        self.last_model_epoch = epoch
        self.last_model_val_loss = current_val_loss

        # Optional rolling epoch-specific "last" checkpoints.
        if self.keep_last_n_epoch_checkpoints > 0:
            epoch_ckpt_name = f"last_model_epoch_{epoch:04d}.zip"
            epoch_ckpt_path = os.path.join(self.save_path, epoch_ckpt_name)
            torch.save(self.net.state_dict(), epoch_ckpt_path)
            self._epoch_checkpoint_history.append(epoch_ckpt_name)
            while len(self._epoch_checkpoint_history) > self.keep_last_n_epoch_checkpoints:
                stale = self._epoch_checkpoint_history.pop(0)
                stale_path = os.path.join(self.save_path, stale)
                if os.path.exists(stale_path):
                    os.remove(stale_path)

        # best model tracking
        if current_val_loss < self.best_val_loss_sentinel:
            self.best_val_loss_sentinel = current_val_loss
            self.best_model_epoch = epoch
            self.best_model_val_loss = current_val_loss
            torch.save(self.net.state_dict(), os.path.join(self.save_path, "best_model.zip"))
            print(f"model saved with best validation loss: {self.best_val_loss_sentinel:0.4f}")

        # Keep metadata up to date on every epoch.
        self._write_checkpoint_metadata()

        # prepare metrics dict for plotting
        metrics: Dict[str, List] = {
            "Training Loss Total": self.train_loss_list_per_epoch,
            "Validation Loss Total": self.valid_loss_list_per_epoch,
        }

        # component curves
        for j in range(n_vars):
            metrics[f"Train: Var. {j}"] = [x[j].item() for x in self.train_comp_loss_per_epoch]
            #metrics[f"valid_var_{j}"] = [x[j].item() for x in self.valid_comp_loss_per_epoch]

        if scheme == "Hybrid":
            metrics.update(
                {
                    "$\\alpha$": self.alpha_list,
                    "$\\beta$": self.beta_list,
                    "Train: Pred. $m_{\\nu}^{2}$": self.train_pred_mass_per_epoch,
                    #"valid_predInvMass": self.valid_pred_mass_per_epoch,
                    "Train: $\\alpha \\cdot $LCL($p_{\\mu}$)": self.train_alpha_LCL_per_epoch,
                    #"valid_alpha*LCL": self.valid_alpha_LCL_per_epoch,
                    "Train: $\\beta \\cdot m_{\\nu}^{2}$": self.train_beta_PM_per_epoch,
                    #"valid_beta*PM": self.valid_beta_PM_per_epoch,
                }
            )

        # one canvas – many lines (train_LCL draws loss.png in _record_epoch instead)
        if plot:
            plot_loss(metrics, self.save_path, logy=False)

        if self._should_early_stop():
            self.early_stop_triggered = True
            self.early_stop_epoch = epoch
            print(
                "Early stopping triggered: validation loss improvement over the "
                f"last {self.early_stopping_window} epoch(s) is below "
                f"{self.early_stopping_min_delta_pct:.4f}%"
            )
            self._write_checkpoint_metadata()
            return True
        return False

    # ------------------------------------------------------------------
    # eval (unchanged)
    # ------------------------------------------------------------------

    def eval(self):
        self.net.to(self.gpu_device)
        best_ckpt = os.path.join(self.save_path, "best_model.zip")
        last_ckpt = os.path.join(self.save_path, "last_model.zip")
        if os.path.exists(best_ckpt):
            model_to_load = best_ckpt
        elif os.path.exists(last_ckpt):
            print("[WARN] best_model.zip not found; falling back to last_model.zip for evaluation.")
            model_to_load = last_ckpt
        else:
            raise FileNotFoundError(
                f"No checkpoint available for eval in {self.save_path}. "
                "Expected best_model.zip or last_model.zip."
            )

        self.net.load_state_dict(
            torch.load(model_to_load, map_location=torch.device("cpu"), weights_only=True)
        )
        self.net.eval()

        trueval, prediction = [], []
        # Make sure eval does not build graphs and uses lean CUDA paths
        with torch.inference_mode():
            for batch in self.testloader:
                vec, sca, msk, tgt = [x.to(self.gpu_device) for x in batch[:4]]
                out = self.net.forward(vec, sca, msk)
                trueval.append(tgt.detach().cpu().numpy())
                prediction.append(out.detach().cpu().numpy())

        np.savez(os.path.join(self.save_path, "result.npz"), trueval=np.concatenate(trueval), prediction=np.concatenate(prediction))

        # save true and predicted target variables to df 
        true_arr = np.concatenate(trueval, axis=0)
        pred_arr = np.concatenate(prediction, axis=0)
        
        target = self.input_d["target"]
        truevars = [f"true_{x}" for x in target]
        predvars = [f"pred_{x}" for x in target]
        
        truevars_df = pd.DataFrame(true_arr, columns=truevars)
        predvars_df = pd.DataFrame(pred_arr, columns=predvars)
        df = pd.concat([truevars_df, predvars_df], axis=1)
        
        # add tag column
        model_phys_name = self.input_d["model_phys_name"]
        df[model_phys_name] = -999999
        
        df.to_csv(os.path.join(self.save_path, "result.csv"), index=False)

        # Raw config "eval_columns" of the test set, row for row in the order of result.npz
        # (the test loader does not shuffle), so predictions can be split by selection, sample
        # and run/subrun/evt.
        extra = getattr(self.testloader.dataset, "extra", None)
        if extra is not None and getattr(extra, "width", 0) > 0:
            extra_path = os.path.join(self.save_path, "test_eval_columns.parquet")
            extra.write_parquet(extra_path)
            print("eval columns of the test set saved to", extra_path)
