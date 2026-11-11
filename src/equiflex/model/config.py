"""
Configuration management for Flex model.

Provides unified configuration loading from YAML files with dataclass validation.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple

from .models import FlexConfig
from .loss import LossConfig


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = './pdbbind_data'
    test_data_dir: Optional[str] = None
    pb_data_dir: Optional[str] = None  # PoseBusters benchmark dir (separate from test split)
    batch_size: int = 8
    systems_per_batch: int = 1         # Distinct systems per GPU step; each gets batch_size timestep copies
    pad_systems: bool = True            # Pad distinct systems into one faster multi-system batch
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    # Train/val split
    val_ratio: float = 0.05        # Fraction of systems for validation (only used if split_csv is None)
    val_batches: int = 100         # Number of validation batches per validation epoch

    # Pre-computed splits CSV (recommended)
    split_csv: Optional[str] = 'data_v2_plinder/splits.csv'

    # Legacy: directory-based splits (deprecated, use split_csv instead)
    split_dir: Optional[str] = None

    # Pre-load all (train+val) systems' tensors into RAM at startup; with
    # Linux fork-based DataLoader workers the cache is shared copy-on-write.
    preload: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    # Optimizer type: 'adamw', 'muon', or 'normuon'
    optimizer_type: str = 'muon'

    # Muon/NorMuon parameters (for 2D hidden weights)
    muon_lr: float = 0.02             # Muon learning rate
    muon_momentum: float = 0.95       # Muon momentum
    muon_weight_decay: float = 1e-2   # Muon weight decay

    # AdamW parameters (for embeddings, biases, output heads, or all params if optimizer_type='adamw')
    adam_lr: float = 3e-4             # AdamW learning rate
    adam_beta1: float = 0.9           # Adam beta1
    adam_beta2: float = 0.95          # Adam beta2
    adam_eps: float = 1e-10           # Adam epsilon
    adam_weight_decay: float = 1e-5   # AdamW weight decay


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler (linear warmup + linear decay)."""
    warmup_steps: int = 1000              # Linear warmup steps
    start_factor: float = 0.01            # lr starts at start_factor * base_lr
    end_factor: float = 0.0               # lr ends at end_factor * base_lr


@dataclass
class EMAConfig:
    """Configuration for Exponential Moving Average."""
    enabled: bool = True
    decay: float = 0.9999


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    # WandB
    use_wandb: bool = True
    wandb_project: str = 'flex'
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Logging frequency
    log_every_n_steps: int = 50

    # Gradient and parameter logging
    watch_model: bool = True                    # Enable wandb.watch for gradients/params
    watch_log: str = 'gradients'                # 'gradients', 'parameters', 'all', or None
    watch_log_freq: int = 1000                  # How often to log gradients/params

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_top_k: int = 3
    checkpoint_every_n_steps: int = 5000


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    max_steps: int = 100000
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: int = 1000
    precision: str = '32'          # '32', '16-mixed', 'bf16-mixed'
    devices: int = 1
    seed: int = 42
    use_compile: bool = False       # Use torch.compile for faster training
    compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    phys_loss_warmup_steps: int = 5000  # Warmup steps for physical loss terms
    num_rmsd_samples: int = 30
    num_steps: int = 100
    timestep_sampler: str = 'logit_normal'  # 'uniform' or 'logit_normal'
    use_config: bool = False  # ConFIG conflict-free gradient update over [flow, physics] groups
    reset_optimizer_state: bool = False  # drop optimizer/scheduler state on resume (param groups changed)


@dataclass
class FullConfig:
    """
    Complete configuration for Flex training.

    Combines all sub-configurations into a single object.
    """
    # Sub-configs
    model: FlexConfig = field(default_factory=FlexConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary."""
        return {
            'model': asdict(self.model),
            'loss': asdict(self.loss),
            'data': asdict(self.data),
            'optimizer': asdict(self.optimizer),
            'scheduler': asdict(self.scheduler),
            'ema': asdict(self.ema),
            'logging': asdict(self.logging),
            'training': asdict(self.training),
        }

    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FullConfig':
        """Create config from nested dictionary."""
        return cls(
            model=FlexConfig(**config_dict.get('model', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            data=DataConfig(**config_dict.get('data', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            scheduler=SchedulerConfig(**config_dict.get('scheduler', {})),
            ema=EMAConfig(**config_dict.get('ema', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
        )

    @classmethod
    def load(cls, path: str) -> 'FullConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(path: str) -> FullConfig:
    """Load configuration from YAML file."""
    return FullConfig.load(path)


def save_config(config: FullConfig, path: str):
    """Save configuration to YAML file."""
    config.save(path)


def get_default_config() -> FullConfig:
    """Get default configuration."""
    return FullConfig()
