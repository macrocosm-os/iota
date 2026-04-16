from loguru import logger
import math
from abc import ABC, abstractmethod

from common import settings as common_settings


class MockOptimizer:
    """Simple optimizer that just manages learning rate for scheduling"""

    def __init__(self, lr):
        self.lr = lr
        self.param_groups = [{"lr": lr}]

    def step(self):
        pass

    def zero_grad(self):
        pass


class BaseScheduler(ABC):
    """Abstract base class for learning rate schedulers"""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def step(self, step_count: int) -> float:
        """Step the scheduler with explicit step count"""
        pass

    def get_last_lr(self) -> float:
        """Get the current learning rate"""
        return self.optimizer.param_groups[0]["lr"]

    def is_initialized(self) -> bool:
        """Check if the scheduler is properly initialized"""
        return self.optimizer is not None and hasattr(self.optimizer, "param_groups")


class LinearLR(BaseScheduler):
    """Pure Python implementation of LinearLR scheduler"""

    def __init__(self, optimizer, start_factor, end_factor, total_iters):
        super().__init__(optimizer)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

    def step(self, step_count: int) -> float:
        if step_count < self.total_iters:
            progress = step_count / self.total_iters
            factor = self.start_factor + (self.end_factor - self.start_factor) * progress
            for group in self.optimizer.param_groups:
                group["lr"] = common_settings.LEARNING_RATE * factor
            return self.optimizer.param_groups[0]["lr"]


class LambdaLR(BaseScheduler):
    """Pure Python implementation of LambdaLR scheduler"""

    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        self.lr_lambda = lr_lambda

    def step(self, step_count: int) -> float:
        factor = self.lr_lambda(step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = common_settings.LEARNING_RATE * factor
        return self.optimizer.param_groups[0]["lr"]


class SequentialLR(BaseScheduler):
    """Pure Python implementation of SequentialLR scheduler"""

    def __init__(self, optimizer, schedulers, milestones):
        super().__init__(optimizer)
        self.schedulers = schedulers
        self.milestones = milestones

    def step(self, global_step_count: int) -> float:
        # Determine which scheduler should be active
        current_scheduler_idx = self._get_scheduler_index(global_step_count)

        # Calculate the step count relative to the current scheduler's phase
        phase_step_count = self._get_phase_step_count(global_step_count, current_scheduler_idx)

        # Step the current scheduler with the phase-relative step count
        if current_scheduler_idx < len(self.schedulers):
            return self.schedulers[current_scheduler_idx].step(phase_step_count)
        else:
            raise ValueError(
                f"Scheduler index out of bounds: {current_scheduler_idx} with global step count: {global_step_count}"
            )

    def _get_scheduler_index(self, global_step_count: int) -> int:
        """Determine which scheduler should be active based on global step count"""
        scheduler_idx = 0
        for milestone in self.milestones:
            if global_step_count >= milestone:
                scheduler_idx += 1
            else:
                break
        return min(scheduler_idx, len(self.schedulers) - 1)

    def _get_phase_step_count(self, global_step_count: int, scheduler_idx: int) -> int:
        """Calculate step count relative to the current scheduler's phase"""
        if scheduler_idx == 0:
            return global_step_count
        else:
            phase_start = self.milestones[scheduler_idx - 1]
            return global_step_count - phase_start


def make_lr_scheduler(optimizer=None) -> BaseScheduler:
    """
    Here are the stages of this scheduler:
    0. linear warm-up 0 → 1 × LRpeak
    1. constant plateau at LRpeak (optional)
    2. macro-cosine × micro-saw-tooth
    3. tail cosine to zero
    """

    # ─── hyper-parameters from settings.py ────────────────────────────
    # TODO: move this to DB: run_state.config
    warm_steps = common_settings.LR_WARMUP_STEPS  # e.g. 3_500
    plateau_steps = common_settings.LR_CONST_STEPS  # e.g.   500
    total_steps = common_settings.TOTAL_TRAIN_STEPS  # e.g. 100_000
    tail_frac = common_settings.LR_TAIL_STEPS_FRAC  # 0.02 (2 %)
    start_fac = common_settings.LR_WARMUP_START_FACTOR  # 0.002
    final_fac = common_settings.LR_FINAL_FACTOR  # 0.10
    cycle_length = common_settings.LR_SAW_CYCLE_LENGTH  # e.g. 10_000
    # if you prefer "N cycles", set cycle_length = decay_steps // N

    tail_steps = int(total_steps * tail_frac)
    decay_steps = total_steps - warm_steps - plateau_steps - tail_steps
    assert decay_steps > 0, "decay phase would be zero/negative"

    # ─── phase-0: linear warm-up 0 → 1 × LRpeak ───────────────────────

    # If optimizer is None, create a simple mock optimizer
    if optimizer is None:
        optimizer = MockOptimizer(common_settings.LEARNING_RATE)

    sched_warm = LinearLR(
        optimizer,
        start_factor=start_fac,
        end_factor=1.0,
        total_iters=warm_steps,
    )

    # ─── phase-1: constant plateau at LRpeak (optional) ───────────────
    sched_plateau = LambdaLR(optimizer, lr_lambda=lambda _: 1.0) if plateau_steps else None

    # ─── phase-2: macro-cosine × micro-saw-tooth  ────────────────────
    def combined_lambda(step):
        """
        step counts from 0 … decay_steps-1 inside the decay phase
        return LR multiplier ∈ [0, 1]
        """
        # ----- macro envelope  LRpeak → final_fac·LRpeak --------------
        macro_p = step / decay_steps
        macro = final_fac + (1.0 - final_fac) * 0.5 * (1 + math.cos(math.pi * macro_p))

        # ----- micro cosine-restart 1 → 0.1 → 1 every cycle_length ----
        cycle_p = (step % cycle_length) / cycle_length
        micro = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_p))
        # micro ∈ [0.1, 1]

        return macro * micro  # overall multiplier

    sched_saw = LambdaLR(optimizer, lr_lambda=combined_lambda)

    # ─── phase-3: tail cosine to zero ─────────────────────────────────
    def tail_lambda(step):
        p = step / tail_steps
        return final_fac * 0.5 * (1 + math.cos(math.pi * p))  # ↘ 0

    sched_tail = LambdaLR(optimizer, lr_lambda=tail_lambda)

    # ─── stitch phases together ──────────────────────────────────────
    schedulers = [sched_warm]
    milestones = [warm_steps]

    if sched_plateau:
        schedulers.append(sched_plateau)
        milestones.append(milestones[-1] + plateau_steps)

    schedulers += [sched_saw, sched_tail]
    milestones += [milestones[-1] + decay_steps]  # (= total)

    lr_scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    logger.info(
        f"LR schedule\n"
        f"  warm-up   : 0–{warm_steps - 1}\n"
        f"  plateau   : {warm_steps}–{warm_steps + plateau_steps - 1}\n"
        f"  saw-tooth : {milestones[-2] - decay_steps}–{milestones[-2] - 1} "
        f"(cycle_length={cycle_length})\n"
        f"  tail      : {milestones[-2]}–{total_steps - 1}"
    )

    return lr_scheduler
