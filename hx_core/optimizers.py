"""Optional lightweight optimisation helpers for HyperCubeX.

These utilities are **not** required for the core engine – they provide a
simple reward-driven search over synaptic weights.  Importing this module is
safe even if users prefer to keep the emergent dynamics untouched.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np

from .workspace import GlobalWorkspace
from .controller import ControllerAssembly
from .assemblies import Assembly, AssemblyStore

from .scheduler import SimpleScheduler
from .connectors import ConnectionInitializer, RandomConnector

__all__ = ["WeightPerturbOptimizer", "ReinforceOptimizer", "PolicyScheduler", "BatchPolicyScheduler"]


class WeightPerturbOptimizer:  # pylint: disable=too-few-public-methods
    """Random weight perturbation with greedy acceptance.

    At each call to :py:meth:`step`, a random connection weight is perturbed
    within ±``step_size``.  If an external routine (e.g. *PolicyScheduler*)
    decides the modification is detrimental, it may revert the change using
    the returned *change tuple*.
    """

    def __init__(self, step_size: float = 0.1, *, min_w: float = -1.0, max_w: float = 1.0) -> None:  # noqa: D401
        self.step_size = float(step_size)
        self.min_w = float(min_w)
        self.max_w = float(max_w)

    # ------------------------------------------------------------------
    def perturb(self, network: "Any") -> Tuple[Any, Any, float, float] | None:  # noqa: D401
        """Apply a small random perturbation and return a *change* tuple.

        The tuple can later be fed to :py:meth:`revert` to undo the change.
        Returns *None* if the network has no connections.
        """
        if not network.connections:
            return None

        src = random.choice(list(network.connections.keys()))
        if not network.connections[src]:
            return None
        tgt = random.choice(list(network.connections[src].keys()))
        old_w = network.connections[src][tgt]
        new_w = float(
            np.clip(
                old_w + random.uniform(-self.step_size, self.step_size),
                self.min_w,
                self.max_w,
            )
        )
        network.connections[src][tgt] = new_w
        return (src, tgt, old_w, new_w)

    @staticmethod
    def revert(network: "Any", change: Tuple[Any, Any, float, float] | None) -> None:  # noqa: D401
        """Revert a previously perturbed weight."""
        if change is None:
            return
        src, tgt, old_w, _ = change
        network.connections[src][tgt] = old_w


class ReinforceOptimizer:  # pylint: disable=too-few-public-methods
    """Policy-gradient style weight updates (REINFORCE)."""

    def __init__(
        self,
        step_size: float = 0.1,
        log_std: float = -2.0,
        *,
        min_w: float = -1.0,
        max_w: float = 1.0,
    ) -> None:  # noqa: D401
        self.lr = float(step_size)
        self.log_std = float(log_std)
        self.min_w = float(min_w)
        self.max_w = float(max_w)
        self._baseline: float | None = None

    # ------------------------------------------------------------------
    def perturb(self, network: "Any") -> Tuple[Any, Any, float, float, float] | None:  # noqa: D401
        if not network.connections:
            return None
        src = random.choice(list(network.connections.keys()))
        if not network.connections[src]:
            return None
        tgt = random.choice(list(network.connections[src].keys()))
        old_w = network.connections[src][tgt]
        std = float(np.exp(self.log_std))
        noise = random.gauss(0.0, std)
        new_w = float(np.clip(old_w + noise, self.min_w, self.max_w))
        network.connections[src][tgt] = new_w
        # Return noise so we can compute gradient later
        return (src, tgt, old_w, new_w, noise)

    # ------------------------------------------------------------------
    def update(self, network: "Any", change, reward: float) -> None:  # noqa: D401
        if change is None:
            return
        src, tgt, old_w, new_w, noise = change
        # Baseline for variance reduction
        if self._baseline is None:
            self._baseline = reward
        # REINFORCE update: w += lr * (R - baseline) * noise / std^2
        std2 = float(np.exp(2 * self.log_std))
        grad = (reward - self._baseline) * noise / std2
        updated = float(np.clip(new_w + self.lr * grad, self.min_w, self.max_w))
        network.connections[src][tgt] = updated
        # Update baseline EMA
        self._baseline = 0.9 * self._baseline + 0.1 * reward

    # Revert does nothing – we keep updates
    @staticmethod
    def revert(network: "Any", change):  # noqa: D401
        return  # no-op


class PolicyScheduler:  # pylint: disable=too-few-public-methods
    """Combine network stepping with greedy weight search.

    Parameters
    ----------
    adapter : hx_adapters.ArcAdapter
        Adapter already *encoded* with the input grid.
    teacher : hx_teachers.RecursiveTeacher | Any
        Teacher providing ``predict`` and ``reward`` methods.
    optimizer : WeightPerturbOptimizer
        Instance controlling weight perturbations.
    target_reward : float
        Early-stop threshold.
    max_ticks : int
        Upper bound on simulation steps.
    eval_interval : int
        Compute reward every *eval_interval* ticks.
    """

    def __init__(
        self,
        *,
        adapter: "Any",
        teacher: "Any",
        optimizer: str | WeightPerturbOptimizer | ReinforceOptimizer | None = None,
        connector: ConnectionInitializer | None = None,
        target_grid: np.ndarray | None = None,
        target_reward: float = 0.9,
        max_ticks: int = 200,
        eval_interval: int = 5,
        logger: "Any" | None = None,
        workspace: GlobalWorkspace | None = None,
        controller: ControllerAssembly | None = None,
        assembly_store: AssemblyStore | None = None,
        assembly_threshold: float | None = None,
    ) -> None:  # noqa: D401
        self.adapter = adapter
        self.teacher = teacher
        # Factory string → instance
        if isinstance(optimizer, str):
            if optimizer.lower() == "reinforce":
                self.optimizer = ReinforceOptimizer()
            else:
                self.optimizer = WeightPerturbOptimizer()
        else:
            self.optimizer = optimizer or WeightPerturbOptimizer()
        self.connector = connector or RandomConnector()
        self.target_reward = float(target_reward)
        self._explicit_target_grid = target_grid
        self.max_ticks = int(max_ticks)
        self.eval_interval = int(eval_interval)
        self.logger = logger

        # Ensure network has connections to optimise
        if not adapter.network.connections:
            self.connector.apply(network=adapter.network, neurons=adapter.neurons)

        self.scheduler = SimpleScheduler(adapter.network)
        self.best_reward: float = -1.0
        self.best_snapshot: Dict[Tuple[Any, Any], float] = {}

        # Warm-up one step so that neurons with energy > threshold spike once
        self.scheduler.step()

        # Cache target grid based on initial activation pattern
        if self._explicit_target_grid is not None:
            self._target_grid = self._explicit_target_grid
        else:
            self._target_grid = self.teacher.predict(self.adapter.decode())

        # Set up global workspace -------------------------------------------------
        if workspace is None:
            self.workspace = GlobalWorkspace(self._target_grid.shape)
        else:
            self.workspace = workspace

        # Pre-activate memory assemblies ----------------------------------
        if assembly_store is not None:
            self.workspace.inject_assemblies(assembly_store)

        # Controller --------------------------------------------------------------
        self.controller = controller if controller is not None else ControllerAssembly()  # Modified default controller

        # Assemblies persistence ----------------------------------------------
        self.assembly_store = assembly_store
        self.assembly_threshold = (
            target_reward if assembly_threshold is None else float(assembly_threshold)
        )

        # Baseline reward before optimisation
        self.best_reward = self.teacher.reward(self.adapter.decode(), self._target_grid)

    # ------------------------------------------------------------------
    def run(self) -> float:  # noqa: D401
        """Run simulation + greedy search until *target_reward* reached."""
        for tick in range(1, self.max_ticks + 1):
            # 1. Optionally perturb a weight each evaluation cycle
            change = None
            if tick % self.eval_interval == 0:
                change = self.optimizer.perturb(self.adapter.network)

            # 2. Advance one simulation step
            self.scheduler.step()

            # 3. Evaluate reward periodically
            if tick % self.eval_interval == 0:
                raw_out = self.adapter.decode()
                _name, cand_grid = self.controller.act(raw_out)
                reward_val = self.teacher.reward(cand_grid, self._target_grid)

                # Submit vote to global workspace --------------------------
                self.workspace.submit(self.adapter, cand_grid, confidence=reward_val)
                winner_grid = self.workspace.decide()
                winner_source = self.workspace.winner().source

                # Logging if available (on global reward)
                if self.logger is not None:
                    global_reward = self.teacher.reward(winner_grid, self._target_grid)
                    self.logger.log(self.adapter.network, reward=global_reward)

                # Reward back-prop only if this adapter wins ---------------
                assembly_reward = reward_val if winner_source is self.adapter else 0.0

                # Update strengths & persistence ---------------------------
                if self.assembly_store is not None:
                    if isinstance(winner_source, Assembly):
                        # Reinforce winning assembly (EMA)
                        winner_source.update_strength(assembly_reward, alpha=0.1)
                    elif winner_source is self.adapter and assembly_reward >= self.assembly_threshold:
                        # Persist new assembly derived from candidate grid
                        mask = cand_grid != 0  # simple criterion: non-zero pixels
                        self.assembly_store.add(
                            Assembly.from_mask(mask, strength=float(assembly_reward))
                        )

                    # Exponential forgetting for others -----------------
                    decay = 0.999  # ~60% retention after 1k wins
                    for asm in self.assembly_store:
                        if asm is not winner_source:
                            asm.strength *= decay
                    # Keep store sorted after updates
                    self.assembly_store._assemblies.sort(key=lambda a: a.strength, reverse=True)

                # Update controller with obtained reward
                self.controller.update(assembly_reward)

                if isinstance(self.optimizer, ReinforceOptimizer):
                    # Apply policy-gradient update
                    self.optimizer.update(self.adapter.network, change, assembly_reward)
                else:
                    if assembly_reward >= self.best_reward:
                        self.best_reward = assembly_reward
                        # Keep change (greedy)
                    else:
                        # Revert change if it did not improve
                        if not isinstance(self.optimizer, ReinforceOptimizer):
                            WeightPerturbOptimizer.revert(self.adapter.network, change)  # type: ignore[arg-type]

            if self.best_reward >= self.target_reward:
                break
        return self.best_reward


class BatchPolicyScheduler:  # pylint: disable=too-few-public-methods
    """Wrapper to run several PolicySchedulers in parallel and aggregate reward.

    Accepts *adapters* and *target_grids* lists of equal length.  Each pair is
    managed by its own PolicyScheduler instance.  Optimisation happens
    independently, then the mean best reward is returned.  This provides a
    minimal batching capability while reusing existing logic.
    """

    def __init__(
        self,
        *,
        adapters: list[Any],
        teacher: "Any",
        optimizer: str | WeightPerturbOptimizer | ReinforceOptimizer | None = None,
        connector: ConnectionInitializer | None = None,
        target_grids: list[np.ndarray] | None = None,
        target_reward: float = 0.9,
        max_ticks: int = 200,
        eval_interval: int = 5,
        logger: "Any" | None = None,
    ) -> None:  # noqa: D401
        if target_grids is not None and len(target_grids) != len(adapters):
            raise ValueError("target_grids length must match adapters")
        # Share a single optimizer instance across schedulers to maintain a common
        # baseline when using REINFORCE.  If *optimizer* is a factory string we
        # turn it into the concrete object once and reuse it.
        if isinstance(optimizer, str):
            if optimizer.lower() == "reinforce":
                shared_opt: WeightPerturbOptimizer | ReinforceOptimizer | None = ReinforceOptimizer()
            elif optimizer.lower() == "random":
                shared_opt = WeightPerturbOptimizer()
            else:
                raise ValueError("Unsupported optimizer")
        else:
            shared_opt = optimizer  # may be None → PolicyScheduler default

        self.schedulers: list[PolicyScheduler] = []
        for idx, adapter in enumerate(adapters):
            tg = None if target_grids is None else target_grids[idx]
            sched = PolicyScheduler(
                adapter=adapter,
                teacher=teacher,
                optimizer=shared_opt,
                connector=connector,
                target_grid=tg,
                target_reward=target_reward,
                max_ticks=max_ticks,
                eval_interval=eval_interval,
                logger=logger,
            )
            self.schedulers.append(sched)

    # ------------------------------------------------------------------
    def run(self) -> float:  # noqa: D401
        rewards = [s.run() for s in self.schedulers]
        return float(np.mean(rewards))
