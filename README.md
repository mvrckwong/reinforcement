# Reinforcement Learning Training Framework

A clean, modular reinforcement learning training and evaluation framework using Ray RLlib and IMPALA algorithm.

## Project Structure

```
src/
├── train.py              # Main training script
├── evaluate.py           # Main evaluation script
├── training/             # Training package
│   ├── config.py         # Algorithm configuration
│   ├── trainer.py        # Training loop logic
│   ├── evaluator.py      # Evaluation logic
│   ├── checkpoint.py     # Checkpoint management (saving)
│   ├── checkpoint_loader.py  # Checkpoint loading
│   └── metrics.py        # Metrics extraction and reporting
├── configs/              # Configuration utilities
│   ├── paths.py          # Path configurations
│   └── storage.py        # Storage configurations
└── utils/                # Utility functions
    └── s3_upload.py      # S3 upload utilities
```

## Features

- ✅ Clean, modular architecture following ArjanCode best practices
- ✅ Separation of concerns with dedicated modules
- ✅ Support for both local and S3 checkpoint storage
- ✅ Comprehensive evaluation with statistics
- ✅ Progress tracking with tqdm
- ✅ Type hints throughout
- ✅ Proper dependency injection

## Usage

### Training

Train an IMPALA agent on CartPole-v1:

```bash
python src/train.py
```

The training script will:
- Train for 100 iterations (configurable)
- Save checkpoints every 10 iterations
- Display training metrics in real-time
- Save final checkpoint at the end

### Evaluation

Evaluate a trained model:

```bash
# Evaluate the latest checkpoint
python src/evaluate.py

# Evaluate a specific checkpoint
python src/evaluate.py --checkpoint path/to/checkpoint

# Evaluate with more episodes
python src/evaluate.py --num-episodes 200

# Render the environment during evaluation
python src/evaluate.py --render

# List available checkpoints
python src/evaluate.py --list-checkpoints
```

The evaluation script will:
- Load the latest (or specified) checkpoint
- Prioritize local checkpoints (even if S3 is configured for training)
- Run evaluation episodes without exploration
- Display progress and statistics
- Show mean, std, min, max for rewards and episode lengths

### Command Line Options

**evaluate.py:**
- `--checkpoint PATH`: Path to specific checkpoint (default: latest)
- `--num-episodes N`: Number of episodes to evaluate (default: 100)
- `--render`: Render the environment during evaluation
- `--list-checkpoints`: List all available checkpoints

## Configuration

### Training Hyperparameters

Edit `src/training/config.py` to modify:
- Environment name
- Number of rollout workers
- Learning rate, gamma, batch size
- Entropy coefficient
- Value function loss coefficient
- And more...

### Checkpoints

Checkpoints are saved to:
- **Local**: `checkpoints/impala_cartpole/YYYYMMDD_HHmmss/`
- **S3**: Optionally uploaded if configured

**Important**: Local checkpoints are ALWAYS saved (even when S3 is configured) to enable evaluation. If S3 is configured, checkpoints are saved locally AND uploaded to S3.

## Environment Variables

Create a `.env` file with:

```env
# Optional: S3 configuration for checkpoint backup
S3_ENDPOINT_URL=your-s3-endpoint
S3_BUCKET_NAME=model
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
```

**Behavior**:
- Without S3: Checkpoints saved locally only
- With S3: Checkpoints saved locally AND uploaded to S3
- Evaluation: Always uses local checkpoints

## Development

The codebase follows clean architecture principles:

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Components receive dependencies via constructors
3. **Type Safety**: Full type hints throughout
4. **Testability**: Modular design makes unit testing easy
5. **Extensibility**: Easy to add new algorithms or environments

### Adding a New Algorithm

1. Add configuration in `training/config.py`
2. Update imports in `train.py` and `evaluate.py`
3. All other code remains unchanged!

## Requirements

- Python 3.10+
- Ray RLlib
- Gymnasium
- tqdm
- pendulum
- python-dotenv
- boto3 (for S3 support)

See `pyproject.toml` for full dependencies.

## License

MIT
