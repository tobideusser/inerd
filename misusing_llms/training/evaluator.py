import inspect
from typing import Dict, List, Any

from misusing_llms.training.metrics import Metric


class Evaluator:
    def __init__(self, metrics: Dict[str, Metric]):
        self.metrics = metrics

    @property
    def metric_names(self) -> List:
        return list(self.metrics.keys())

    @classmethod
    def from_config(cls, **evaluator_params):

        metrics = {}
        for metric_name, metric_kwargs in evaluator_params.items():
            if metric_kwargs:
                metrics[metric_name] = Metric.from_config(type_=metric_name, **metric_kwargs)
            else:
                metrics[metric_name] = Metric.from_config(type_=metric_name)

        return cls(metrics=metrics)

    def update(self, batch_output: Dict[str, Any]):

        for metric_name, metric in self.metrics.items():
            expected_arguments = inspect.signature(metric.update).parameters.keys()
            metric_kwargs = {arg: batch_output[arg] for arg in expected_arguments if arg in batch_output}

            metric.update(**metric_kwargs)

    def compute(self, reset: bool = False):
        metrics = {}
        for metric in self.metrics.values():
            metrics = {**metrics, **metric.compute_wrapped(reset=reset)}

        return metrics
