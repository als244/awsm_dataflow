# AdaWS: Adaptive Working Set Strategy for High Throughput Transformer Training Under Memory Constraints

## Steps

### 0. Clone the Repository

```bash
git clone https://github.com/als244/awsm_dataflow.git
```

### 1. Set Up Python Environment

Use a Python environment with PyTorch and Flash Attention 2 or Flash Attention 3 installed.

### 2. Install Helper Modules

Install the helper Python modules (matmul dispatcher and transmission scheduler):

```bash
make
```

### 3. Launch the Dashboard

Run the dashboard server to view training progress:

```bash
python dashboard/dashboard_server.py --port 8501
```

This will start a web server at `http://localhost:8501`. If you are working on a remote server, run this command and then create a port forward to access the dashboard from local machine. This can be done with the following command:

```bash
ssh -N -f -L 8501:<server_hostname>:8501 <username>@<remote_hostname>
# An example command for use with Princeton Della Cluster (in local shell): 
# 'ssh -N -f -L 8501:della-k17g3:8501 as1669@della.princeton.edu'
# Then open http://localhost:8501 in your browser
```


### 4. Run Training

#### a) Benchmark Training

Run a benchmark training loop using randomly generated fixed-length sequence data:

```bash
python bench_train.py [OPTIONS]
```

##### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run_name` | str | `default_run` | Run name (displayed in dashboard) |
| `--model_choice` | str | `llama3_8B` | Model choice key from `model_dims.json` |
| `--seq_len` | int | `8192` | Sequence length |
| `--seqs_per_step` | int | `64` | Sequences per step |
| `--max_steps` | int | `10` | Max training steps (0 = unlimited) |
| `--max_gpu_mem_gb` | float | `None` | Max GPU memory in GB (none = detect available capacity) |
| `--max_host_mem_gb` | float | `None` | Max host memory in GB (none = detect available capacity) |
| `--use_muon` | bool | `True` | Use Muon optimizer (`--use_muon false` to disable) |

##### Example

```bash
# Run with defaults
python bench_train.py

# Custom configuration using maximum of 20GB of GPU memory and 120GB of host memory
python bench_train.py --run_name hello_flops --model_choice llama3_8B --seq_len 65536 --seqs_per_step 5 --max_gpu_mem_gb 20 --max_host_mem_gb 120
```

#### b) FineWeb Run

First, download the FineWeb dataset:

```bash
python fineweb.py
```

Then run training on the downloaded data using the same command-line arguments as the benchmark:

```bash
python train.py [OPTIONS]
```

### Model Choices

Default model configurations are defined in `model_dims.json`. Available choices:

- `nanogpt_124M`
- `llama3_8B`
- `olmoe_7Bx1B`
- `dense_15B`
- `sparse_16Bx3B`
- `qwen3_32B`
- `qwen3_30Bx3B`

These are **randomly initialized** models that share the same architecture dimensions as their namesakes — they do not use pretrained weights.

### Training Hyperparameters

Training hyperparameters are defined within the `train.py` script; edit the script to change them.