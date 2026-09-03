# Trained surrogates

DPNN weights: heater volts and input port in, the four photodiode volts out. `learn.dpnn`
is the model, `learn.prune` the training loop, `learn.train_hw --from-table` the loader
that fitted these from stored data with no hardware.

| | `mrunal_28k` | `table_25c` |
|---|---|---|
| source | `mrunal/28k_Data_points .xlsx` | `pic_data/sessions/2026-08-27/raw_transfers_fresh.json` |
| measured | 2026-08-26 | 2026-08-27 |
| samples | 28,000 (7,000 states x 4 ports) | 540 (135 states x 4 ports) |
| channels with a feature | DAC 0-5 | DAC 0-5, 9, 10 |
| parameters after pruning | 2,753 | 3,295 |
| held-out R^2 | **0.9756** | 0.8896 |

Per detector, `mrunal_28k` over its own 28,000 rows: PD0 0.9989, PD1 **0.9071**, PD2 0.9985,
PD3 0.9984. PD1 is not a worse detector here, it is a narrower one -- its readings span
0.14-0.43 V where the others span 0.00-0.70, so an equal absolute residual is a larger
share of a smaller variance.

Fifty times the data is worth 0.09 R^2, and the smaller fit is the one that is current.
Neither dominates: `mrunal_28k` is the better model of the chip *as it was*, and predates
both the present clamp table and the characterization in `pic_data/calib.json`, so what it
does not know is every phase the chip has drifted through since. `table_25c` is thin enough
that its held-out split is 81 samples, but it is the only one that saw DAC 9 and 10 move.

Two things neither of them saw. No laser telemetry was recorded with the spreadsheet and
only the power and temperature with the capture, so the telemetry block sits at the centre
of its fixed scale in `learn.dpnn._SCALES` and normalises to zero -- an absent input, not
an invented one. And 11,481 of the spreadsheet's rows sit above this tree's present ceiling
on DAC 1 and DAC 4, because it was swept to a uniform 3.0 V before the per-channel clamps
existed. They are kept: they are real measurements, and only the clamp has moved since.
That is what `pic.config.drive_volts_raw` is for -- clipping them into the feature matrix
would report a voltage the chip never saw.

    from learn.dpnn import load_ckpt
    model, norm, buf, meta = load_ckpt("pic_data/surrogate/mrunal_28k")   # buf is None

`buffer.npz` is deliberately absent. It is the training data, not the model -- 5.8 MB of
numbers already tracked in the sheet beside it -- and only `--resume` needs it. `calib.json`
is absent for the same reason: with `--steps 0` it is `pic_data/calib.json` unchanged.

Regenerate either, in about five minutes and one minute respectively:

    ./do train --from-table "mrunal/28k_Data_points .xlsx" --steps 0 --epochs 400 --out runs/mrunal
    ./do train --from-table pic_data/sessions/2026-08-27/raw_transfers_fresh.json --steps 0 --epochs 600 --out runs/table

`--steps 0` fits the network only. Dropping it also refits the 52-parameter physics model
in `learn.unitary_fit`, which is minutes of gradient descent through the twin per restart
against seconds of backprop here, and nothing is written until it finishes.
