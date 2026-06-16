# DSTIGFN

Official PyTorch implementation of **DSTIGFN: Dynamic Spatio-Temporal Interactive Graph Fusion Network for Traffic Flow Prediction**, accepted by *IEEE Transactions on Intelligent Transportation Systems*.

DSTIGFN is a traffic flow forecasting model designed to capture dynamic spatial dependencies and multi-scale temporal patterns in road sensor networks. Given the previous 12 traffic observations, the model predicts the next 12 time steps in a direct multi-step forecasting manner.

## Paper

**DSTIGFN: Dynamic Spatio-Temporal Interactive Graph Fusion Network for Traffic Flow Prediction**  
Shiqi Zhang, Zhen Liu, Yuzhuang Pian, Jialong Qian, Yonghong Liu, and H. Oliver Gao  
*IEEE Transactions on Intelligent Transportation Systems*, 2026  
DOI: [10.1109/TITS.2026.3696998](https://doi.org/10.1109/TITS.2026.3696998)

## Highlights

- **Temporal periodic encoding**: uses learnable intra-day and day-of-week embeddings to represent traffic periodicity.
- **Trend-preserving parity decomposition**: splits traffic sequences into odd and even subsequences while preserving temporal trends.
- **Dynamic graph fusion**: combines adaptive graph structures and supra-Laplacian dynamic graphs to model evolving spatial dependencies.
- **Spatio-temporal interactive fusion**: progressively fuses odd/even subsequence features across multiple temporal scales.
- **Direct multi-step prediction**: forecasts all 12 future horizons in one forward pass to reduce error accumulation.

## Repository Structure

```text
DSTIGFN-master-main/
|-- config_file/
|   |-- PeMS03_DSTIGFN.conf
|   `-- PeMS04_DSTIGFN.conf
|   |-- PeMS07_DSTIGFN.conf
|   `-- PeMS08_DSTIGFN.conf
|-- data/
|   |-- PeMS03/
|   |-- PeMS04/
|   |-- PeMS07/
|   `-- PeMS08/
|-- lib/
|   |-- data_processing.py
|   |-- engine.py
|   |-- ranger21.py
|   `-- utils.py
|-- model/
|   `-- DSTIGFN.py
|-- train.py
`-- README.md
```

## Environment

The experiments in the paper were implemented with PyTorch. A typical environment is:

```bash
conda create -n dstigfn python=3.8 -y
conda activate dstigfn

pip install numpy pandas
pip install torch
```

Install the PyTorch version that matches your CUDA driver from the official PyTorch website. The paper reports experiments using PyTorch 1.11.0 on an NVIDIA GeForce RTX 3090 GPU.

## Data

The model expects each dataset as an `.npz` file with a `data` array:

```text
data shape: [num_time_steps, num_nodes, num_features]
```

The current code uses the first channel as the target traffic flow and can append two temporal features:

- time of day
- day of week

Expected dataset statistics from the paper:

| Dataset | Nodes | Sampling interval | Input length | Prediction length |
| --- | ---: | --- | ---: | ---: |
| PeMS03 | 358 | 5 minutes | 12 | 12 |
| PeMS04 | 307 | 5 minutes | 12 | 12 |
| PeMS07 | 883 | 5 minutes | 12 | 12 |
| PeMS08 | 170 | 5 minutes | 12 | 12 |

Current repository note:

- `data/PeMS03/PEMS03.npz` and `data/PeMS08/PEMS08.npz` are present.
- `data/PeMS04/` and `data/PeMS07/` currently contain CSV/text files only. Convert them to `.npz` before training with the provided config files.
- The provided config files use uppercase paths such as `./data/PEMS08/PEMS08.npz`, while the current folders are named `PeMS08` and `PeMS04`. On case-sensitive file systems, update `data_file_path` before running.
- `train.py` uses `dataSet_name = "PeMS08"` to build the default config path and experiment folder name. When running another dataset, pass `--config` explicitly and update `dataSet_name` if you want the output folder name to match the dataset.

Example:

```ini
[Data]
dataset_name = PeMS08
data_file_path = ./data/PeMS08/PEMS08.npz
num_of_vertices = 170
history_seq_len = 12
future_seq_len = 12
train_ratio = 0.6
valid_ratio = 0.2
steps_per_day = 288
TOD = True
DOW = True
```

## Training

Train DSTIGFN on PeMS08:

```bash
python train.py --config ./config_file/PeMS08_DSTIGFN.conf
```

Train DSTIGFN on PeMS04 after preparing `PEMS04.npz` and fixing the path:

```bash
python train.py --config ./config_file/PeMS04_DSTIGFN.conf
```

Main training options are configured in `config_file/*.conf`:

| Option | Description |
| --- | --- |
| `history_seq_len` | Number of historical time steps used as input |
| `future_seq_len` | Number of future time steps to predict |
| `num_of_vertices` | Number of road sensors/nodes |
| `batch_size` | Training batch size |
| `nb_chev_filter` | Hidden channel size |
| `learning_rate` | Initial learning rate |
| `dropout` | Dropout rate |
| `alph` | Inter-layer supra-Laplacian coupling coefficient |
| `gama` | Top-k graph sparsification ratio |
| `runs` | Number of independent training runs |
| `device_id` | CUDA device id |

## Outputs

Training results are saved under:

```text
experiments/{dataset}_{model}/{timestamp}/session_{run_id}/
```

Each session contains:

- `best_model.pth`: best checkpoint selected during training
- `train.csv`: training and validation metrics
- `test.csv`: horizon-wise test metrics and average results

Logs are saved under:

```text
experiments/{dataset}_{model}/{timestamp}/logs/
```

The reported metrics include:

- MAE
- RMSE
- MAPE
- WMAPE

## Model Components

The main implementation is in `model/DSTIGFN.py`.

- `TemporalEmbedding`: learnable time-of-day and day-of-week embeddings.
- `AGSG`: adaptive graph structure generation.
- `MHSG`: supra-Laplacian graph generation for cross-temporal dependencies.
- `DGGC`: dynamic generation graph convolution with fused and sparsified attention.
- `STIF`: spatio-temporal interactive fusion for odd/even subsequences.
- `STGIF`: stacked multi-scale interaction and temporal order restoration.
- `DSTIGFN`: full forecasting network.

## Citation

If you find this repository useful, please cite:

```bibtex
@ARTICLE{11563915,
  author={Zhang, Shiqi and Liu, Zhen and Pian, Yuzhuang and Qian, Jialong and Liu, Yonghong and Gao, H. Oliver},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={DSTIGFN: Dynamic Spatio-Temporal Interactive Graph Fusion Network for Traffic Flow Prediction},
  year={2026},
  volume={},
  number={},
  pages={1-14},
  keywords={Traffic flow prediction;spatio-temporal heterogeneity;parity decomposition;interactive fusion},
  doi={10.1109/TITS.2026.3696998}
}
```

## Acknowledgement

The datasets used in this project are from the Caltrans Performance Measurement System (PeMS). The optimizer implementation is based on Ranger21.

## License

Please check the repository license before using this code for commercial or redistribution purposes.
