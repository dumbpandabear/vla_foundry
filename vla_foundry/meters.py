import logging
import math

import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self):
        self.stats = {
            "batch_time": AverageMeter(),
            "data_time": AverageMeter(),
            "forward_time": AverageMeter(),
            "backward_time": AverageMeter(),
            "optim_step_time": AverageMeter(),
            "sync_time": AverageMeter(),
            "loss": AverageMeter(),
            "grad_norm": AverageMeter(),
        }

    def update_and_log_state(
        self,
        cfg,
        batch_size,
        batch_num_tokens,
        batch_count,
        num_batches_per_checkpoint,
        step,
        dataloader,
        lr,
        checkpoint_num,
    ):
        num_samples = batch_count * batch_size * cfg.distributed.world_size
        samples_per_checkpoint = dataloader.dataloader.num_samples
        percent_complete = 100.0 * batch_count / num_batches_per_checkpoint

        # Throughput stats (samples / tokens per second).
        samples_per_second = batch_size * cfg.distributed.world_size / self.stats["batch_time"].val
        samples_per_second_per_gpu = batch_size / self.stats["batch_time"].val
        tokens_per_second = batch_num_tokens * cfg.distributed.world_size / self.stats["batch_time"].val
        tokens_per_second_per_gpu = batch_num_tokens / self.stats["batch_time"].val

        loss_str = f"Loss: {self.stats['loss'].avg:.3f}"
        sample_digits = math.ceil(math.log(dataloader.dataloader.num_samples + 1, 10))
        logging.info(
            f"Train Checkpoint: {checkpoint_num} "
            f"[{num_samples:>{sample_digits}}/{samples_per_checkpoint} "
            f"({percent_complete:.0f}%)] "
            f"{loss_str} "
            f"Data (t): {self.stats['data_time'].avg:.3f} "
            f"Batch (t): {self.stats['batch_time'].avg:.3f}, "
            f"{samples_per_second:#g}/s, "
            f"{samples_per_second_per_gpu:#g}/s/gpu "
            f"LR: {lr:5f} "
        )

        # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
        self.state = {
            "loss": self.stats["loss"].val,
            "grad_norm": self.stats["grad_norm"].val,
            "data_time": self.stats["data_time"].val,
            "batch_time": self.stats["batch_time"].val,
            "forward_time": self.stats["forward_time"].val,
            "backward_time": self.stats["backward_time"].val,
            "optim_step_time": self.stats["optim_step_time"].val,
            "sync_time": self.stats["sync_time"].val,
            "samples_per_second": samples_per_second,
            "samples_per_second_per_gpu": samples_per_second_per_gpu,
            "tokens_per_second": tokens_per_second,
            "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
            "lr": lr,
            # ``step`` has already been incremented after this optimizer step.
            "tokens": step * cfg.hparams.global_batch_size * cfg.data.seq_len,
            "samples": step * cfg.hparams.global_batch_size,
            "expected_steps_epoch": dataloader.dataloader.num_batches,
            "seen_steps_epoch": batch_count,
            "checkpoint": checkpoint_num,
            "checkpoint_progress_pct": percent_complete,
            "global_step": step,
        }
        if torch.cuda.is_available():
            self.state.update(
                {
                    "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "gpu_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            )

        if cfg.wandb:
            import wandb

            log_dict = {f"train/{name}": val for name, val in self.state.items()}
            log_dict["tokens"] = self.state["tokens"]
            log_dict["samples"] = self.state["samples"]
            log_dict["global_step"] = step
            wandb.log(log_dict, step=step)

        self.reset()

    def reset(self):
        for meter in self.stats:
            self.stats[meter].reset()
