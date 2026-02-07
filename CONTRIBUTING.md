# Contributing to AsyncPP

Thank you for your interest in contributing to the Asynchronous Pipeline Parallel project!

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.5.1+
- CUDA 12.6+ (for GPU experiments)
- At least 8 GPUs (for running the full pipeline)

### Setup

```bash
git clone https://github.com/PluralisResearch/AsyncPP.git
cd AsyncPP
pip install -r requirements.txt
```

### Running Experiments

The main entry point is `run.bash`, which launches both the asynchronous method and GPipe baseline:

```bash
bash run.bash
```

This script configures an 8-stage pipeline on a WikiText-103 language modeling task. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nnodes` | 8 | Number of pipeline stages |
| `batch` | 8 | Mini-batch size |
| `epochs` | 50 | Training epochs |
| `lr` | 3e-4 | Learning rate |
| `momentum` | 0.99 | Nesterov momentum (our method) |

## Project Structure

```
AsyncPP/
├── main_with_runtime.py   # Async pipeline parallel training (our method)
├── sync_main.py           # Synchronous baselines (GPipe, 1F1B)
├── run.bash               # Launch script for experiments
├── data_utils.py          # Dataset loading utilities
├── models/
│   └── gptn/              # GPT model partitioned across pipeline stages
├── runtime/
│   ├── runtime.py         # Core pipeline runtime (scheduling, forward/backward)
│   ├── communication.py   # Distributed communication primitives
│   └── runtime_utilities.py
└── optim/
    ├── adamw.py           # AdamW optimizer
    └── nadamw.py          # NAdamW optimizer (Nesterov variant)
```

## How to Contribute

### Reporting Issues

Open a GitHub issue with:
- A clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (Python version, PyTorch version, GPU setup)

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Ensure your code passes linting: `flake8 . --select=E9,F63,F7,F82`
5. Commit with a descriptive message
6. Push and open a PR against `main`

### Code Style

- Follow PEP 8 guidelines
- Maximum line length: 120 characters
- Use descriptive variable names
- Add comments for non-obvious logic

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ajanthan2025asyncpp,
  title={Nesterov Method for Asynchronous Pipeline Parallel Optimization},
  author={Ajanthan, Thalaiyasingam and Ramasinghe, Sameera and Zuo, Yan and Avraham, Gil and Long, Alexander},
  journal={ICML},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
