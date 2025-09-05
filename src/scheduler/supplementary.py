from typing import List, Optional, Iterable, Any, Union
from math import lcm
import warnings

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scheduler.task_models import Task, PeriodicTask, Job

# hyperparameters
arrow_dy = 40
arrow_dx = 10

# TODO: default lcm works for integers only
# TODO: integrate huperperiod into PlotManager
def get_periodic_tasks_hyperperiod(taskload: List[Task]) -> float:
    """Compute the hyperperiod of a list of periodic tasks."""

    periods = [task.period for task in taskload if isinstance(task, PeriodicTask)]
    if not periods:
        return 0.0
    return lcm(*periods)


def add_hyperperiod_lines(fig: go.Figure, T: float, interval: float):
    for i in range(interval // T):
        fig.add_vline(
            x=T*(i+1),
            line=dict(color="black", width=1),
            opacity=0.4
        )
    return fig


def random_hex_color():
    n = int(np.random.randint(0, 0x1000000))
    return f"#{n:06X}"


class PlotManager:
    def __init__(self, workload: List[Job] = []):
        self.fig = go.Figure()
        self.colors = {}
        self.workload = workload
    
    def plot_backlog(self, workload: List[Job] = None, exe_trace_width: int = 4, arrival_trace_width: int = 2):
        if not workload:
            workload = self.workload
        
        fig = go.Figure()
        colormap = {}
        for job in workload:
            if job.task.name not in colormap:
                colormap[job.task.name] = random_hex_color()

        self.colormap = colormap
        simulation_stack = workload.copy()
        simulation_stack.reverse()
        traces = []
        current_time = 0.0
        current_backlog = 0.0 # workload backlog
        # TODO: use deque, as pop(0) is O(n)
        queue = [] # task backlog
        
        while(simulation_stack):
            # skipping quiet time till the nearest job arrival
            if ((not queue) and simulation_stack):
                if current_time <= simulation_stack[-1].release_time:
                    current_time = simulation_stack[-1].release_time
                    new_job = simulation_stack.pop()
                    queue.append(new_job)

                    # vertivcal backlog increment
                    fig.add_annotation(
                        x=current_time, y=current_backlog, text=f"{job.task.name}-{job.job_index}",
                        showarrow=True, arrowhead=2, ax=arrow_dx, ay=-arrow_dy,
                        bgcolor=colormap[new_job.task.name]
                    )
                    traces.append(go.Scatter(
                        x=[current_time, current_time],
                        y=[current_backlog, current_backlog + new_job.execution_time],
                        mode="lines",
                        line=dict(color=colormap[new_job.task.name], width=arrival_trace_width, dash="dash"),
                        name=f"{new_job.task.name}-{new_job.job_index} arrival",
                    ))
                    current_backlog += new_job.execution_time
            
            job = queue.pop(0)
            
            execution_time_remaining = job.execution_time
            
            # looking for the nearest interruption
            while (simulation_stack and (current_time + execution_time_remaining > simulation_stack[-1].release_time)):
                # plotting until the interruption
                traces.append(go.Scatter(
                    x=[current_time, simulation_stack[-1].release_time],
                    y=[current_backlog, current_backlog - (simulation_stack[-1].release_time - current_time)],
                    mode="lines",
                    line=dict(color=colormap[job.task.name], width=exe_trace_width),
                    name=f"{job.task.name}",
                ))
                job.service_start = current_time
                processed = simulation_stack[-1].release_time - current_time
                execution_time_remaining -= processed
                current_backlog -= processed
                current_time = simulation_stack[-1].release_time
                
                # vertical backlog increments // < is redundant actually? it should always be ==
                while (simulation_stack and simulation_stack[-1].release_time <= current_time):
                    arrived_job = simulation_stack.pop()
                    queue.append(arrived_job)
                    fig.add_annotation(
                        x=current_time, y=current_backlog, text=f"{arrived_job.task.name}-{job.job_index}",
                        showarrow=True, arrowhead=2, ax=arrow_dx, ay=-arrow_dy,
                        bgcolor=colormap[arrived_job.task.name]
                    )
                    traces.append(go.Scatter(
                        x=[current_time, current_time],
                        y=[current_backlog, current_backlog + arrived_job.execution_time],
                        mode="lines",
                        line=dict(color=colormap[arrived_job.task.name], width=arrival_trace_width, dash="dash"),
                        name=f"{arrived_job.task.name}-{arrived_job.job_index} arrival",
                    ))
                    current_backlog += arrived_job.execution_time
            
            # plotting the tail without interruptions
            if execution_time_remaining >= 0:
                job.service_start = job.service_start or current_time
                traces.append(go.Scatter(
                    x=[current_time, current_time + execution_time_remaining],
                    y=[current_backlog, current_backlog - execution_time_remaining],
                    mode="lines",
                    line=dict(color=colormap[job.task.name], width=exe_trace_width),
                    name=f"{job.task.name}",
                ))
                current_time += execution_time_remaining
                current_backlog -= execution_time_remaining   
                fig.add_annotation(
                    x=current_time, y=current_backlog, text=f"{job.task.name}-{job.job_index} done",
                    showarrow=True, arrowhead=2, ax=arrow_dx, ay=arrow_dy,
                    bgcolor=colormap[job.task.name]
                )
                
            else:
                Warning("Execution time remaining is negative, something went wrong.")
                
        fig.add_traces(traces) 
        fig.update_layout(
            title="System Backlog Over Time",
            xaxis_title="Time",
            yaxis_title="Backlog",
            showlegend=True
        )
        return fig

    def plot_task_timeline(
        self,
        tasks_names_reference: List[str],
        workload: List[Job] = None,
        relative_to_arrival: bool = True,
        title: str | None = None,
        sort_reverse: bool = True,
        colormap: dict[str, Any] = None
        ):
        if not workload:
            workload = self.workload
            
        if not colormap:
            colormap = self.colormap
        
        task_workload: List[Job] = [j for j in workload if (j.task.name in tasks_names_reference)]
        
        if not task_workload:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(0.5, 0.5, "No jobs for selected task", ha="center", va="center")
            ax.axis("off")
            plt.show()
            return

        task_workload.sort(key=lambda j: j.release_time, reverse=sort_reverse)

        fig, ax = plt.subplots(figsize=(12, max(2.5, 0.7 * len(task_workload))))

        for i, j in enumerate(task_workload):
            rel = float(j.release_time)
            exe = float(j.execution_time)
            D   = float(j.relative_deadline)
            s0  = j.service_start if j.service_start is not None else rel

            if relative_to_arrival:
                shift = rel
                x_arrival  = 0.0
                x_deadline = D
                bar_start  = float(s0 - shift)
            else:
                x_arrival  = rel
                x_deadline = rel + D
                bar_start  = float(s0)

            ax.vlines(x_arrival,  i - 0.4, i + 0.4, linestyles="dashed")
            ax.vlines(x_deadline, i - 0.4, i + 0.4, linestyles="dotted")
            ax.broken_barh([(bar_start, exe)], (i - 0.35, 0.7), color=colormap[j.task.name] if colormap else None)
            ax.text(x_arrival,  i + 0.42, f"arr={rel:.2f}", ha="left", va="bottom", fontsize=8)
            ax.text(x_deadline, i - 0.42, "D", ha="left", va="top", fontsize=8)

            if isinstance(j.task, PeriodicTask):
                P = j.task.period
                if (relative_to_arrival):
                    if (j.job_index == 0):
                        x_period = P
                        ax.axvline(x_period, color=colormap[j.task.name] if colormap else 'black', linewidth=2.0, alpha=0.25)
                else:
                    x_period = rel + P
                    ax.axvline(x_period, color=colormap[j.task.name] if colormap else 'black', linewidth=2.0, alpha=0.25)
                if (j.job_index == 0):
                    ax.axvline(j.task.offset, color=colormap[j.task.name] if colormap else 'black', linewidth=2.0, alpha=0.25)

        ax.set_yticks(range(len(task_workload)))
        ax.set_yticklabels([f"job {j.job_index} ({j.task.name})" for j in task_workload])
        ax.set_xlabel("Time (relative to arrival)" if relative_to_arrival else "Time (absolute)")



        if title is None:
            title = f"Service timeline for task '{getattr(task_workload[0].task, 'name', '?')}'"
        ax.set_title(title)

        ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
        fig.tight_layout()
        plt.show()
