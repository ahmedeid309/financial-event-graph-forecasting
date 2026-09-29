# Event Knowledge Graph Embeddings for Stock Forecasting

Code for the master's thesis *Improving Financial Time-Series Forecasting with LLM-Derived
Event Knowledge Graph Embeddings* (Ahmed Eid, Institute of Parallel and Distributed Systems,
University of Stuttgart, 2026).

A large language model extracts typed financial events from news articles. The events form an
event knowledge graph, and a temporal graph neural network condenses the graph into one vector
per company and day. TimeXer fuses these vectors with ten days of prices to forecast the next
trading day's adjusted close. The experiments compare the graph representation with price-only
forecasting, with direct text embeddings of the same articles, and with the sentiment scores of
the FNSPID dataset. They also re-run the FNSPID study's own Transformer configuration.

```
FNSPID news ──► 1 extraction ──► 2 graph ──► 3 graph encoder ──► daily company vectors ──┐
                (Qwen3.5-35B-A3B)  (BGE, clustering)  (temporal GNN)                     │
FNSPID prices ──► price correction ──────────────────────────────────────────────────────┤
                                                                                          ▼
                    4 TimeXer: PRICE vs GRAPH · 4b TEXT · 4c encoders · 5 ablation · 7 SENTIMENT
                    6, 8a-8e FNSPID task and Transformer · 9 previous-day news
```

## Repository layout

| Path | Contents |
|---|---|
| `data_prep/` | `select_news.py`: the thesis corpus from the FNSPID news archive |
| `financial_ekg/` | Event extraction (step 1) and graph construction (step 2) |
| `gnn/` | Supervision table, temporal graph encoder, daily representations, text and sentiment features, joins to prices, price correction (`fix_price_splice.py`), tests |
| `timexer_overlay/` | The changes to TimeXer; see its README |
| `fnspid_transformer/` | Replication of the FNSPID Transformer (the authors' code is fetched, not included) |
| `analysis/` | Statistics and tables for each experiment |
| `scripts/` | One SLURM launcher per step, `0_setup_external.sh`, `run_chain.sh` |
| `tools/` | `checkpoint_manifest.py` |
| `checkpoints_sha256.txt` | SHA-256 of every trained model file behind the thesis results |

## Setup

All commands run from the repository root.

**1. Environment** (Python 3.12, CUDA 11.8)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.23
pip install -r requirements.txt
```

The launchers do not activate an environment. Activate it before `sbatch`; SLURM passes it on
to the job. If llama-cpp-python loads the CUDA libraries of the pip `nvidia-*` packages, add
their `lib` directories to `LD_LIBRARY_PATH`.

**2. Third-party code** (TimeXer and the FNSPID code and files)

```bash
bash scripts/0_setup_external.sh
```

**3. Models**

```bash
mkdir -p models
# The extraction model: Qwen3.5-35B-A3B, Q4_K_M GGUF, placed at
#   models/Qwen3.5-35B-A3B-Q4_K_M.gguf   (22,016,023,168 bytes)
sha256sum models/Qwen3.5-35B-A3B-Q4_K_M.gguf
# expected 3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375
hf download BAAI/bge-large-en-v1.5 --revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 \
    --local-dir models/bge-large-en-v1.5
```

**4. Data** (FNSPID, huggingface.co/datasets/Zihan1004/FNSPID)

```bash
mkdir -p dataset
wget -P dataset https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
wget -P dataset https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
unzip -q dataset/full_history.zip -d dataset      # per-ticker price files in dataset/full_history/
python data_prep/select_news.py --archive_csv dataset/nasdaq_exteral_data.csv \
    --output_csv dataset/news_2018_2023.csv       # 36,992 articles
python gnn/fix_price_splice.py                    # -> dataset/full_history_fixed/
```

`fix_price_splice.py` corrects the adjustment break of 6 July 2020 in the five focal companies'
price files; its docstring gives the evidence, the procedure and the factors.

## Running the experiments

Submit each launcher from the repository root, e.g. `sbatch scripts/4_train_timexer.sh`. Every
launcher states its purpose, inputs and outputs in its header. Its runs are written out one by
one, so each launcher is also the complete record of its configuration.
Log messages call the steps "stages".

| Step | Launcher | What it does | Thesis |
|---|---|---|---|
| 0 | `0_setup_external.sh` | TimeXer and FNSPID code and files | – |
| 1 | `1_extract_events.sh` | Event extraction, two partitions per job (29 partitions in total) | 3.2 |
| 2 | `2_build_graph.sh` | Event knowledge graph | 3.3 |
| 3 | `3_train_gnn.sh` | Graph encoder (3 seeds), daily representations, the shared split | 3.4, 3.5, 4.4.1 |
| 4 | `4_train_timexer.sh` | PRICE and GRAPH, 5 companies × 5 seeds | 3.6, 4.3.1 |
| 4b | `4b_train_text_baseline.sh` | TEXT64 and TEXT1024 controls | 3.7.2, 4.3.2 |
| 4c | `4c_encoder_variance.sh` | GRAPH with each of the three encoders | 3.8.4, 4.4.2 |
| 5 | `5_ablate_fusion.sh` | Fusion ablation, 8 variants | 4.5 |
| 6 | `6_matched_fnspid.sh` | PRICE and GRAPH on the FNSPID three-day task | 4.6.3 |
| 7 | `7_train_sentiment_arm.sh` | SENTIMENT (FNSPID scores) | 3.7.3, 4.3.3 |
| 8a | `8a_fnspid_transformer_check.sh` | FNSPID Transformer on the FNSPID stocks (reproduction check) | 4.6.4 |
| 8b | `8b_fnspid_transformer_exact.sh` | FNSPID protocol on the thesis companies | 4.6.4 |
| 8c | `8c_fnspid_transformer_forecast.sh` | FNSPID Transformer on the three-day task | 4.6.4 |
| 8d | `8d_sentiment_matched_task.sh` | Sentiment score on the three-day task, both models | 4.6.4 |
| 8e | `8e_fnspid_transformer_onestep.sh` | FNSPID Transformer on the one-step task | 4.6.4 |
| 9 | `9_previous_day_news.sh` | One-step task with previous-day news only | – |

**Order.** 1 → 2 → 3 → 4. Steps 4b, 4c, 5, 6 and 7 need step 4; 8c needs 6; 8d needs 6 and 7;
8e needs 4 and 7; 9 needs 4, 7 and 8e. `scripts/run_chain.sh` submits 3 → 4 → 4b as one
dependent chain. Before 8a and 8b, build their input files once:

```bash
python fnspid_transformer/build_data.py --repo_data_dir external/fnspid_data_eecdcdb \
    --sentiment_dir fnspid_sentiment \
    --embeddings_csv gnn_outputs_shared/daily_ticker_embeddings.csv --out_dir fnspid_transformer
```

## Analysis

| After step | Command | Result |
|---|---|---|
| 4, 4b, 4c, 5 | `python analysis/analyze_final.py` | Paired results of PRICE, GRAPH and TEXT; encoder sensitivity, which first checks that encoder 2022 reproduces step 4; fusion ablation |
| 4, 4b, 7 | `python analysis/analyze_dm_tests.py` | Diebold–Mariano tests on the daily test losses |
| 4, 6 | `python analysis/analyze_fnspid.py` | Both tasks in the FNSPID metric set |
| 8a | `cd fnspid_transformer && python check_their_weights.py --weights_dir their_weights_5873ff8` | The authors' released weights scored on their data |
| 8a–8c | `python analysis/analyze_fnspid_transformer.py` | Reproduction of FNSPID Table 3 and the exact replication |
| 8d | `python analysis/analyze_fnspid_grid.py` then `python fnspid_transformer/make_latex_tables.py` | Three-day grid |
| 8e | `python analysis/analyze_fnspid_grid_onestep.py` then `python fnspid_transformer/make_latex_tables_onestep.py` | One-step grid |
| 9 | `python analysis/analyze_fnspid_grid_prevday.py` then `python fnspid_transformer/make_latex_tables_prevday.py` | Previous-day grid |

## Reproducibility notes

- **One split.** Step 3 fixes the chronological split. Every later step reads its two boundary
  dates from the exported representations, so no model is selected on another model's test days.
- **Appended results.** TimeXer appends to its result files. Remove or archive old result files
  before rerunning a step; the launchers of steps 5, 8d and 9 refuse to start when outputs exist.
- **Determinism.** With fixed inputs the TimeXer runs reproduce exactly. Step 4c checks this for
  25 runs. The graph encoder sums messages in a nondeterministic order on the GPU, so it is
  trained with three seeds.
- **Caches** are keyed on content: text embeddings on the article text, training examples on a
  fingerprint of the graph.
- **Model files.** `checkpoints_sha256.txt` lists the SHA-256 of all 618 trained model files: 3
  graph encoders, 550 TimeXer runs and 65 FNSPID Transformer runs.
- `gnn/tests` holds unit tests for the encoder (`python -m pytest gnn/tests -q`).

## Third-party code and data

- **TimeXer** (github.com/thuml/TimeXer) carries no license file, so it is not redistributed.
  `scripts/0_setup_external.sh` fetches commit `7601190` and applies `timexer_overlay/`.
- **FNSPID** code and files (github.com/Zdong104/FNSPID_Financial_News_Dataset, CC BY-NC 4.0) are
  fetched at commits `5873ff8` (code, weights) and `eecdcdb` (data files). The news archive and
  price histories come from the FNSPID dataset on Hugging Face.
- **Qwen3.5-35B-A3B** and **BGE-large-en-v1.5** are used under their own licenses.

## Citation

```bibtex
@mastersthesis{eid2026ekg,
  author = {Ahmed Eid},
  title  = {Improving Financial Time-Series Forecasting with LLM-Derived Event Knowledge Graph Embeddings},
  school = {Institute of Parallel and Distributed Systems, University of Stuttgart},
  year   = {2026}
}
```
