from typing import List, Optional

import numpy as np

from scheduler.task_models import Task, PeriodicTask, Job

class Uniprocessor:
    def __init__(self, policy, taskload: List[Task]):
        self.policy = policy
        self.taskload = taskload
        self.workload: List[Job] = []
        self.sampled_jobload_per_task = {}
        
    def get_workload_until(self, T: float, rng: Optional[np.random.Generator] = None) -> List[Job]:
        """Generate all jobs with release_time <= T from the taskload."""
        jobs: List[Job] = []
        for task in self.taskload:
            task_jobs: List[Job] = task.sample_jobs_until(T, rng)
            # self.sampled_jobload_per_task[task] = task_jobs
            jobs.extend(task_jobs)
            
        if (self.policy == "FIFO"):
            jobs.sort(key=lambda j: j.release_time)
        else:
            raise NotImplementedError(f"Policy {self.policy} not implemented.")

        self.workload = jobs
        return jobs
    


    def process(self, until: float) -> List[Job]:
        """Simulate processing jobs until time `until`."""
        raise NotImplementedError()
        