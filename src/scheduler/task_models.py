from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from scheduler.stochastic_source import StochasticSource, IID, Discrete, AR1, ColoredNoise

class Policy(Enum):
    FIFO = "FIFO"
    FP = "Fixed Priority"
    
    
# TODO: Implement more policies
# TODO: eq=False doesn't seem to be a good practice, somehow solve 
@dataclass#(eq=False)
class Task:
    name: str = "Task"
    priority: int = 1
    relative_deadline: float = 10


@dataclass#(eq=False)
class PeriodicTask(Task):
    period: float = 10.0
    offset: float = 0.0
    base_execution_time: float = 1.0
    execution_deviation_distribution: Optional[StochasticSource] = None
    jitter_distribution: Optional[StochasticSource] = None

    def sample_jobs_until(self, T: float, rng: Optional[np.random.Generator] = None) -> List[Job]:
        """Generate jobs with release_time <= T."""
        rng = rng or np.random.default_rng()
        n = int(np.floor((T - self.offset) / self.period)) + 1

        if (T < self.offset) or (n <= 0):
            return []

        jit = (self.jitter_distribution.sample(n, rng)
               if self.jitter_distribution else np.zeros(n))
        dev = (self.execution_deviation_distribution.sample(n, rng)
               if self.execution_deviation_distribution else np.zeros(n))
    
        base_times = self.offset + self.period * np.arange(n)
        releases = base_times + jit
        ground_truth_exec_times = np.maximum(0.0, self.base_execution_time + dev)

        jobs: List[Job] = []
        for (i, (release, exe)) in enumerate(zip(releases, ground_truth_exec_times)):
            if release <= T:
                jobs.append(Job(
                    task=self,
                    release_time=float(release),
                    execution_time=float(exe),
                    relative_deadline=self.relative_deadline,
                    job_index=i,
                ))

        jobs.sort(key=lambda j: j.release_time)
        return jobs
        
    
    
# TODO: sporadic task


@dataclass
class Job:
    task: Task
    release_time: float
    execution_time: float
    relative_deadline: float
    service_start: float = None
    job_index: Optional[int] = None
