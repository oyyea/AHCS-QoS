# AHCS-QoS

**Attribute Hypergraphs with Compositional Supervision for QoS Prediction**

## Project Structure

```text
.
├── main.py                   # Training and evaluation entry point
├── configs/
│   ├── overall.yaml          # Shared settings and search parameters
│   ├── rt.yaml               # Response time settings
│   └── tp.yaml               # Throughput settings
├── models/
│   ├── hyperinter.py         # Full AHCS-QoS model
│   └── hgnn_qos.py           # Attribute-hypergraph-only model
├── untils/
│   ├── Configurator.py       # Configuration loading
│   ├── dataset.py            # Data loading, splitting, and hypergraphs
│   ├── trainer.py            # Training, validation, and evaluation
│   └── logger.py             # Logging
├── data/
│   └── dataset1/             # QoS matrices and contextual metadata
└── log/                      # Generated logs
```

## Dependencies

The training code imports:

- PyTorch
- DeepHypergraph (`dhg`)
- NumPy
- pandas
- PyYAML
- scikit-learn

Install compatible versions of PyTorch and DeepHypergraph for your environment. 

CUDA is used when available; otherwise, the code falls back to CPU.

## Running Experiments

Run commands from the project root.

### Full Model

Response time:

```bash
python main.py --data rt --model ours --gpu_id 0
```

Throughput:

```bash
python main.py --data tp --model ours --gpu_id 0
```

### Cold-Start Experiments

The default shared configuration enables both flags:

```yaml
filter_out_cod_start_users: True
filter_out_cod_start_wss: True
```

The splitter randomly holds out 169 users and 300 services. Their invocation records are excluded from training and validation.

The evaluator reports MAE and RMSE for:

- **Normal:** neither endpoint belongs to the held-out sets.
- **UC:** only the user belongs to a held-out set.
- **SC:** only the service belongs to a held-out set.
- **DC:** both endpoints belong to held-out sets.

### Observation-Density Experiments

Disable explicit node holdout:

```yaml
filter_out_cod_start_users: False
filter_out_cod_start_wss: False
train_ratio: [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10]
```

The sampled records are divided into 90% training and 10% validation; the remaining records form the test set. For example, a density of `0.10` gives approximately 9% training, 1% validation, and 90% testing.

With cold-start holdout enabled, the sampling density instead applies to records remaining after held-out-node invocations are removed.
