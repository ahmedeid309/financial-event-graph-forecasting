# Changes to TimeXer

These files replace or extend the authors' implementation of TimeXer
(github.com/thuml/TimeXer, commit `76011909357972bd55a27adba2e1be994d81b327`).
`scripts/0_setup_external.sh` clones that commit into `TimeXer/` and copies these files over it.
Compare a file with its upstream version to see the exact change, e.g.
`git -C TimeXer diff -- models/TimeXer.py` after setup.

| File | Change |
|---|---|
| `models/TimeXer.py` | Graph fusion. The trailing news block of the input (`--graph_feature_count` columns) passes a LayerNorm and an adapter (`--graph_adapter_dim`). It reaches the global price token through a dedicated cross-attention (`--graph_fusion_mode attention`), or joins the exogenous variables (`concat`). A tanh gate that starts at `--graph_gate_init` 0 scales the graph residual; `--no_graph_gate` removes the gate. The graph modules are created after the price pathway, so paired runs with the same seed initialise all shared price parameters identically. `freeze_price_pathway()` leaves only the graph modules trainable. |
| `exp/exp_long_term_forecasting.py` | Loading of a price-only parent (`--pretrained_checkpoint`), freezing of the price pathway, validation of the initial model as a candidate checkpoint (`--validate_initial_model`), and logging of the gate values with each result. |
| `data_provider/data_loader.py` | A chronological split by date (`--train_end`, `--val_end`), inherited from the exported graph representations, instead of row fractions; the `gnn_*` columns are placed as one block immediately before the target; `drop(..., axis=1)` for current pandas. |
| `utils/tools.py` | A constant learning-rate schedule (`--lradj constant`, used by every run) and `np.inf` for NumPy 2. |
| `run.py` | Command-line options for the changes above; `--seed` is applied after parsing (upstream seeds before the arguments are read, so the option had no effect); an append-only result file (`--result_file`); optional split plots. |
| `make_price_only_csv.py` (new) | Writes the price-only twin of a joined file, with identical rows. |
| `summarize_ablation.py` (new) | Summarises one company's fusion-ablation result file (step 5). |
