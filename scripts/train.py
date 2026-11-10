"""Training script for Flex model."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.plugins.environments import SLURMEnvironment

from equiflex.model import FlexTrainer, EMACallback, load_config
from equiflex.model.loader import get_train_val_dataloaders, get_test_dataloader
from equiflex.model.trainer import RMSDEvaluationCallback
from equiflex.model.config import (
    FullConfig, DataConfig, OptimizerConfig, SchedulerConfig,
    EMAConfig, LoggingConfig, TrainingConfig
)
from equiflex.model.loss import LossConfig
from equiflex.model.models import FlexConfig

# Allow config classes to be unpickled (required for PyTorch 2.6+ with weights_only=True)
torch.serialization.add_safe_globals([
    FullConfig, DataConfig, OptimizerConfig, SchedulerConfig,
    EMAConfig, LoggingConfig, TrainingConfig, LossConfig, FlexConfig
])


def main():
    parser = argparse.ArgumentParser(description='Train Flex model')
    parser.add_argument('--config', type=str, default='configs/equiflex_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed from config')
    parser.add_argument('--wandb-run-id', type=str, default=None,
                        help='Resume logging to existing wandb run ID')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.seed is not None:
        config.training.seed = args.seed

    pl.seed_everything(config.training.seed)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main_process = local_rank == 0
    ngpus = torch.cuda.device_count()
    if ngpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    else:
        strategy = 'auto'

    # Set float32 matmul precision for tensor cores
    torch.set_float32_matmul_precision('high')

    preload_data = getattr(config.data, 'preload', False)
    train_loader, val_loader, train_systems, val_systems, _ = get_train_val_dataloaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        val_ratio=config.data.val_ratio,
        seed=config.training.seed,
        log_dir=config.logging.checkpoint_dir,
        val_batches=config.data.val_batches,
        split_csv=config.data.split_csv,
        split_dir=getattr(config.data, 'split_dir', None),
        preload=preload_data,
        systems_per_batch=getattr(config.data, 'systems_per_batch', 1),
        pad_systems=getattr(config.data, 'pad_systems', True),
    )

    test_loader = None
    test_systems_for_callback = None
    try:
        test_loader = get_test_dataloader(
            test_data_dir=config.data.test_data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )
        if test_loader is not None:
            test_systems_for_callback = [d.name for d in test_loader.dataset.system_dirs]
    except (FileNotFoundError, ValueError) as e:
        print(f'Could not create test dataloader: {e}')

    # PoseBusters benchmark: losses (pb/*_loss) and RMSD (pb_rmsd/*) at every validation
    pb_loader = None
    pb_data_dir = getattr(config.data, 'pb_data_dir', None)
    if pb_data_dir is not None:
        pb_loader = get_test_dataloader(
            test_data_dir=pb_data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

    print(f'Training with config: {args.config}')
    print(f'Train systems: {len(train_loader.dataset)}')
    print(f'Val systems: {len(val_loader.dataset)}')
    if test_loader:
        print(f'Test systems: {len(test_loader.dataset)}')
    print(f'Batch size: {config.data.batch_size}')
    print(f'Optimizer: {config.optimizer.optimizer_type}')
    lo = config.loss
    print(f'Physics: warmup {config.training.phys_loss_warmup_steps} steps, unrolls {lo.num_unrolls}, '
          f'weights bond {lo.bond_length_weight}/{lo.bond_angle_weight} vdw {lo.vdw_weight}/{lo.vdw_active_weight}/'
          f'{lo.vdw_max_weight} form {getattr(lo, "vdw_form", "mean")} buffer {lo.vdw_buffer} rr {getattr(lo, "vdw_rec_rec_ratio", "-")} '
          f'hbond {getattr(lo, "hbond_weight", 0)}@{getattr(lo, "hbond_native_cutoff", 3.5)} torsion {lo.torsion_weight} chiral {getattr(lo, "chiral_weight", 0)}')
    print(f'Precision: {config.training.precision}')
    print(f'Seed: {config.training.seed}')

    # Create model (before logger so wandb.watch can reference it)
    trainer_module = FlexTrainer(config)
    trainer_module.test_dataloader_ref = test_loader
    trainer_module.pb_dataloader_ref = pb_loader
    print('Loaded FlexTrainer')

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{config.logging.checkpoint_dir}/{config.logging.wandb_run_name or "flex"}',
        filename='{step}-{val/flow_loss:.4f}',
        save_top_k=config.logging.save_top_k,
        # val/flow_loss, not val/total_loss: the total includes the physics terms,
        # so changing their weights mid-run re-scales it and every later value
        # would rank below the stored best-k, silently stopping all saves.
        monitor='val/flow_loss',
        mode='min',
        save_last=True,
        enable_version_counter=False,  # overwrite last.ckpt instead of writing last-v1.ckpt
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
    )
    callbacks.append(checkpoint_callback)
    # Unconditional snapshots (no monitor, keep all) for offline evaluation and rollback.
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_callback.dirpath,
        filename='periodic-{step}',
        monitor=None,
        save_top_k=-1,
        save_last=False,
        every_n_train_steps=5000,
    ))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    model_summary = ModelSummary(max_depth=2)
    callbacks.append(model_summary)

    if config.ema.enabled:
        ema_callback = EMACallback(decay=config.ema.decay)
        callbacks.append(ema_callback)
        print(f'EMA enabled (decay={config.ema.decay})')

    if is_main_process:
        if test_loader is not None:
            rmsd_callback = RMSDEvaluationCallback(
                train_systems=train_systems,
                val_systems=val_systems,
                test_systems=test_systems_for_callback,
                data_dir=config.data.data_dir,
                test_data_dir=config.data.test_data_dir,
                pb_data_dir=getattr(config.data, 'pb_data_dir', None),
                num_samples=config.training.num_rmsd_samples,
                num_steps=config.training.num_steps,
                eval_every_n_val=1,
            )
            callbacks.append(rmsd_callback)
        else:
            print('No test loader found, skipping RMSD evaluation callback')

    # Resolve wandb run ID and checkpoint path for auto-resume
    run_name = config.logging.wandb_run_name or 'flex'
    ckpt_dir = Path(f'{config.logging.checkpoint_dir}/{run_name}')
    run_id_file = ckpt_dir / '.run_id'

    wandb_run_id = args.wandb_run_id
    ckpt_path = args.resume

    if wandb_run_id is None and run_id_file.exists():
        wandb_run_id = run_id_file.read_text().strip()
        if is_main_process:
            print(f'Auto-resuming wandb run: {wandb_run_id}')

    if ckpt_path is None:
        # Newest of last*.ckpt: Lightning writes last-v1.ckpt, last-v2.ckpt ... when
        # the checkpoint callback's state key changed across a resume, leaving the
        # unversioned last.ckpt stale.
        last_ckpts = sorted(ckpt_dir.glob('last*.ckpt'), key=lambda p: p.stat().st_mtime)
        if last_ckpts:
            ckpt_path = str(last_ckpts[-1])
            if is_main_process:
                print(f'Auto-resuming from: {ckpt_path}')

    if wandb_run_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb_run_id = f'{run_name}_{timestamp}'

    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(wandb_run_id)

    # Logger: wandb, or a CSV file when the cluster has no outbound network.
    # The CSV logger deletes an existing metrics.csv in its directory, so each
    # (re)launch gets its own version directory under logs/<run>/ and nothing
    # from before a resume is lost.
    logger = None
    if not config.logging.use_wandb and is_main_process:
        launch = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger = CSVLogger(save_dir='logs', name=run_name, version=f'{wandb_run_id}_launch{launch}')
        print(f'CSV metrics: {logger.log_dir}/metrics.csv')
    if config.logging.use_wandb and is_main_process:
        logger = WandbLogger(
            project=config.logging.wandb_project,
            name=config.logging.wandb_run_name,
            entity=config.logging.wandb_entity,
            id=wandb_run_id,
            resume='allow',
            save_dir='logs',
            config=config.to_dict(),
            log_model=False,
        )
        logger.experiment.watch(
            trainer_module.model, log='all', log_freq=config.logging.watch_log_freq,
        )

    # SLURM auto-requeue
    plugins = []
    if SLURMEnvironment.detect():
        plugins.append(SLURMEnvironment(auto_requeue=True))
        if is_main_process:
            print(f'SLURM detected — auto-requeue enabled (run_id={wandb_run_id})')

    # Create trainer. Manual optimization (use_config) forbids Lightning
    # managing gradient_clip_val / accumulate_grad_batches — we apply both
    # manually in FlexTrainer._config_backward_and_step.
    trainer_kwargs = dict(
        max_steps=config.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy=strategy,
        plugins=plugins,
        precision=config.training.precision,
        val_check_interval=config.training.val_check_interval,
        check_val_every_n_epoch=None,
        # One logger row per optimizer step: Lightning counts microbatches here.
        log_every_n_steps=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=is_main_process,
        enable_model_summary=False,
        sync_batchnorm=True,
    )
    if not getattr(config.training, 'use_config', False):
        trainer_kwargs['gradient_clip_val'] = config.training.gradient_clip_val
        trainer_kwargs['accumulate_grad_batches'] = config.training.accumulate_grad_batches

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(
        trainer_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

if __name__ == '__main__':
    main()
