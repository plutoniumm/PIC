"""Dynamically pruned MLP: structured neuron pruning during training, aimed at the
middle ground between the dense DNN (`scripts/learn_compare.py:run_mlp`, accurate but
needs a full retrain) and the tree model (`BlackBoxForward`, cheap to retrain but a
lower R^2 ceiling — see `scripts/retrain_curve.py`).

Every `prune_every_pct` of training progress, the weakest neurons in each hidden layer
are cut and training continues on the smaller layer; survivors keep their learned
weights and the remaining epochs let them absorb what the cut neurons were doing. No
dropout: the shrinking width is the regularizer, and dropout's stochastic zeroing would
corrupt the neuron-importance signal used to decide what to cut.

Neuron "value" = L2 norm of its *outgoing* weights (how much it can still affect the
next layer / the output), not activation magnitude — a neuron can fire all day but if
every downstream weight from it is ~0 it isn't contributing anything, and this stays a
cheap deterministic score across arbitrary batches.

Default prune rule (`criterion="std"`) cuts anything outside 2 standard deviations,
dynamically sized per phase rather than a fixed fraction — but the 2-sigma cut runs in
*log* space, not raw score space: these scores are norms, so they're non-negative and
right-skewed (roughly log-normal), and "mean - 2*std" computed directly on them is
almost always negative and prunes nothing. Taking the mean/std of log(score) first, then
mapping the threshold back with exp(), is the version of "2 sigma" that actually cuts a
real (and phase-dependent) fraction each time — tight score spreads prune little, wide
spreads prune a lot. `criterion="percentile"` (fixed fraction per phase) is kept for
comparison.

Skip connections: past `skip_depth_threshold` hidden layers, layer j's pre-activation
gets a linear-projected addition from layer j-skip_every's output (ResNet-style, every
`skip_every` layers) — thin-and-deep is where a pure feedforward stack degrades hardest,
and pruning pushes width down over training, so a wide-shallow net that starts under the
threshold never gets skips, but a deliberately deep one does automatically. The
projections are resized in lockstep whenever either endpoint gets pruned.

VERBATIM COPY of 6x6/src/prune.py -- the pruning core is chip-independent and
proven; what is reduced for the 4x4 is how it is used (`learn.dpnn`), not this.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

ACTS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


class DynamicPrunedMLP(nn.Module):
    def __init__(self, din, dout, hidden=(512, 256, 128), activation="relu", min_neurons=8,
                 skip_depth_threshold=3, skip_every=2):
        super().__init__()
        act = ACTS[activation]
        self.din, self.dout, self.min_neurons = din, dout, min_neurons
        self.act_cls = act
        sizes = [din] + list(hidden)
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.acts = nn.ModuleList(act() for _ in hidden)
        self.out = nn.Linear(sizes[-1], dout)

        self.skip_every = skip_every
        self.use_skips = len(hidden) > skip_depth_threshold
        self.skip_proj = nn.ModuleDict()
        if self.use_skips:
            for j in range(skip_every, len(self.layers)):
                src_w = self.layers[j - skip_every].out_features
                dst_w = self.layers[j].out_features
                self.skip_proj[str(j)] = nn.Linear(src_w, dst_w, bias=False)

    def forward(self, x):
        h = x
        hs = []
        for i, (lin, a) in enumerate(zip(self.layers, self.acts)):
            pre = lin(h)
            key = str(i)
            if self.use_skips and key in self.skip_proj:
                pre = pre + self.skip_proj[key](hs[i - self.skip_every])
            h = a(pre)
            hs.append(h)
        return self.out(h)

    def widths(self):
        return [lin.out_features for lin in self.layers]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _drop_mask(score: np.ndarray, frac: float, criterion: str, std_k: float) -> np.ndarray:
        if criterion == "percentile":
            thresh = np.percentile(score, frac * 100)
        elif criterion == "std":
            log_s = np.log(score + 1e-12)
            thresh = np.exp(log_s.mean() - std_k * log_s.std())
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        return score <= thresh

    @torch.no_grad()
    def prune_step(self, frac: float = 0.15, criterion: str = "std", std_k: float = 2.0,
                    max_drop_frac: float = 0.5):
        """Drop each hidden layer's weakest neurons — count is dynamic under
        criterion="std" (whatever falls outside `std_k` log-sigma that phase), fixed
        under "percentile". `max_drop_frac` caps a single phase at removing at most half
        a layer's surviving neurons, so an unusually spread-out score distribution can't
        gut a layer in one step. Returns a list of (n_before, n_dropped) per hidden layer."""
        report = []
        for i, lin in enumerate(self.layers):
            next_lin = self.layers[i + 1] if i + 1 < len(self.layers) else self.out
            score = next_lin.weight.norm(dim=0).cpu().numpy()  # one score per output neuron of `lin`
            if self.use_skips:
                dst_key = str(i + self.skip_every)
                if dst_key in self.skip_proj:
                    skip_score = self.skip_proj[dst_key].weight.norm(dim=0).cpu().numpy()
                    score = np.sqrt(score**2 + skip_score**2)  # a neuron feeding only the skip path still counts
            n = lin.out_features
            drop_mask = self._drop_mask(score, frac, criterion, std_k)
            drop_n = min(int(drop_mask.sum()), int(max_drop_frac * n))
            keep_n = max(self.min_neurons, n - drop_n)
            if keep_n >= n:
                report.append((n, 0))
                continue
            idx = np.sort(np.argsort(-score)[:keep_n])
            idx_t = torch.as_tensor(idx, dtype=torch.long)

            new_lin = nn.Linear(lin.in_features, keep_n)
            new_lin.weight.copy_(lin.weight[idx_t])
            new_lin.bias.copy_(lin.bias[idx_t])
            self.layers[i] = new_lin

            new_next = nn.Linear(keep_n, next_lin.out_features)
            new_next.weight.copy_(next_lin.weight[:, idx_t])
            new_next.bias.copy_(next_lin.bias)
            if i + 1 < len(self.layers):
                self.layers[i + 1] = new_next
            else:
                self.out = new_next

            if self.use_skips:
                dst_key = str(i + self.skip_every)
                if dst_key in self.skip_proj:  # layer i's output shrank -> proj's input columns
                    old = self.skip_proj[dst_key]
                    new = nn.Linear(keep_n, old.out_features, bias=False)
                    new.weight.copy_(old.weight[:, idx_t])
                    self.skip_proj[dst_key] = new
                src_key = str(i)
                if src_key in self.skip_proj:  # layer i's pre-act shrank -> proj's output rows
                    old = self.skip_proj[src_key]
                    new = nn.Linear(old.in_features, keep_n, bias=False)
                    new.weight.copy_(old.weight[idx_t])
                    self.skip_proj[src_key] = new

            report.append((n, n - keep_n))
        return report


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update: W_eff = W + (alpha/r)·B·A.

    B starts at zero, so the wrapped layer is initially the identity of the base layer —
    a LoRA fine-tune begins exactly where the warm-started net left off, and only the
    r·(in+out) adapter entries move. Used to test whether post-drift adaptation needs the
    whole pruned net retrained or just a low-rank correction on top of it."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float | None = None,
                 train_bias: bool = False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        if train_bias and base.bias is not None:
            base.bias.requires_grad_(True)
        r = max(1, min(rank, base.in_features, base.out_features))
        self.r = r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = (alpha if alpha is not None else r) / r

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(model: DynamicPrunedMLP, rank: int = 4, train_bias: bool = False):
    """Wrap every Linear in a trained DynamicPrunedMLP with a LoRA adapter and freeze the
    rest. Returns (model, n_trainable)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for i, lin in enumerate(model.layers):
        model.layers[i] = LoRALinear(lin, rank, train_bias=train_bias)
    model.out = LoRALinear(model.out, rank, train_bias=train_bias)
    if model.use_skips:
        for k, proj in list(model.skip_proj.items()):
            model.skip_proj[k] = LoRALinear(proj, rank, train_bias=False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_train


def _r2(y, yh):
    y = np.asarray(y)
    return 1 - ((y - yh) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)


def train_pruned(
    model: DynamicPrunedMLP,
    Xtr, Ytr_norm, Xval, Yval_raw, y_mean, y_std,
    total_epochs: int = 200,
    warmup_pct: int = 20,
    prune_every_pct: int = 10,
    prune_frac: float = 0.15,
    criterion: str = "std",
    std_k: float = 2.0,
    bs: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 1,
    verbose: bool = True,
):
    """Train with a progressive-pruning schedule. The first `warmup_pct` of epochs run
    at full width untouched — pruning by weight norm before the net has had a chance to
    converge just measures noise, not which neurons are actually weak ("early kill").
    After warmup, prunes every `prune_every_pct` of total_epochs up to 90% (e.g.
    warmup=20, every=10 -> checkpoints at 20,30,...,90%, so the last 10% of epochs is a
    fixed-architecture fine-tune the best-val checkpoint is picked from).

    Each inter-prune phase tracks its own best-val-R^2 checkpoint and *prunes from that
    checkpoint*, not from whatever epoch the schedule happens to land on — pruning
    importance scores (and the weights survivors carry forward) come from the
    best-generalizing snapshot of the current width, not a state that's already started
    overfitting. Without this, one overfit phase compounds into the next since the
    smaller net inherits its predecessor's weights."""
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr_norm, dtype=torch.float32)
    Xv = torch.tensor(Xval, dtype=torch.float32)
    n = len(Xt)

    prune_epochs = {
        int(total_epochs * p / 100) for p in range(warmup_pct, 100, prune_every_pct)
    }
    prune_epochs.discard(0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []  # (epoch, val_r2_mean, n_params, widths)
    phase_best = (-1e9, None)  # best val checkpoint within the current-width phase

    def evaluate():
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy() * y_std + y_mean
        model.train()
        return float(_r2(Yval_raw, pv).mean())

    for ep in range(1, total_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i : i + bs]
            opt.zero_grad()
            loss = ((model(Xt[j]) - Yt[j]) ** 2).mean()
            loss.backward()
            opt.step()

        if ep % eval_every == 0 or ep == total_epochs or ep in prune_epochs:
            vr = evaluate()
            history.append((ep, vr, model.n_params(), model.widths()))
            if vr > phase_best[0]:
                phase_best = (vr, {k: v.clone() for k, v in model.state_dict().items()})

        if ep in prune_epochs:
            if phase_best[1] is not None:
                model.load_state_dict(phase_best[1])
            report = model.prune_step(frac=prune_frac, criterion=criterion, std_k=std_k)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            phase_best = (-1e9, None)
            if verbose:
                dropped = [d for _, d in report]
                print(
                    f"  [prune @ep{ep:3d}/{total_epochs}] widths -> {model.widths()}  "
                    f"params={model.n_params()}  dropped={dropped}"
                )

    if phase_best[1] is not None:
        model.load_state_dict(phase_best[1])
    return model, history


def finetune_fixed(
    model: DynamicPrunedMLP,
    Xtr, Ytr_norm, Xval, Yval_raw, y_mean, y_std,
    epochs: int = 60,
    bs: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Warm-start fine-tune at fixed (already-pruned) width — no further pruning. This
    is the cheap side of "retrain": a small net, starting from weights that already
    solved a related session, adapting to a handful of fresh samples from a new one."""
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Yt = torch.tensor(Ytr_norm, dtype=torch.float32)
    Xv = torch.tensor(Xval, dtype=torch.float32)
    n = len(Xt)
    bs = max(1, min(bs, n))
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    best = (-1e9, None)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i : i + bs]
            opt.zero_grad()
            loss = ((model(Xt[j]) - Yt[j]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).numpy() * y_std + y_mean
        vr = float(_r2(Yval_raw, pv).mean())
        if vr > best[0]:
            best = (vr, {k: v.clone() for k, v in model.state_dict().items()})
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best[0]
