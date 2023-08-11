import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Set

import numpy as np
from torchmetrics import BLEUScore

from inerd.utils import entity_string_to_entity_dataclass


def compute_metrics(eval_pred):
    pass


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
        self.saved_observations = 0

    @classmethod
    def from_config(cls, type_: str, *args, **kwargs):
        try:
            class_ = METRICS[type_]
        except KeyError:
            raise KeyError(f'Metric "{type_}" is not implemented.')

        if "entity_set" not in inspect.signature(class_).parameters.keys():
            del kwargs["entity_set"]
        return class_(name=type_, *args, **kwargs)

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update internal metric states."""
        raise NotImplementedError

    def compute_wrapped(self, reset: bool) -> Dict[str, Any]:
        """Compute and return the metric. Optionally also call `self.reset()`."""
        result = self.compute()
        if reset:
            self.reset()
        return result

    @abstractmethod
    def compute(self) -> Union[Dict[str, Any], Any]:
        """Compute and return the metric."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any accumulators or internal state."""
        raise NotImplementedError


class NERF1(Metric):
    def __init__(self, name: str, entity_set: Set[str]):
        super().__init__(name=name)

        self.entity_set = entity_set

        self.ground_truth_entities: List[List[dict]] = []
        self.entity_strings_predicted: List[str] = []

    def update(self, ground_truth_entities: List[List[dict]], entity_string_predicted: List[str]):
        self.ground_truth_entities.extend(ground_truth_entities)
        self.entity_strings_predicted.extend(entity_string_predicted)
        self.saved_observations += len(entity_string_predicted)

    def compute(self):
        assert len(self.ground_truth_entities) == len(self.entity_strings_predicted)

        statistics = {ent: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for ent in self.entity_set}
        clf_report = {}

        predicted_entities = [entity_string_to_entity_dataclass(es) for es in self.entity_strings_predicted]

        # Count TP, FP and FN per type
        for prediction, ground_truth in zip(predicted_entities, self.ground_truth_entities):
            for entity_type in self.entity_set:
                pred_ents = {ent.words for ent in prediction if ent.type_ == entity_type}
                gt_ents = {" ".join(ent["words"]) for ent in ground_truth if ent["type_"] == entity_type}
                statistics[entity_type]["support"] += len(gt_ents)
                statistics[entity_type]["tp"] += len(pred_ents & gt_ents)
                statistics[entity_type]["fp"] += len(pred_ents - gt_ents)
                statistics[entity_type]["fn"] += len(gt_ents - pred_ents)

        # Compute per entity Precision / Recall / F1 / Support
        for entity_type in statistics.keys():
            if statistics[entity_type]["tp"]:
                precision = (
                    100
                    * statistics[entity_type]["tp"]
                    / (statistics[entity_type]["fp"] + statistics[entity_type]["tp"])
                )
                recall = (
                    100
                    * statistics[entity_type]["tp"]
                    / (statistics[entity_type]["fn"] + statistics[entity_type]["tp"])
                )
            else:
                precision, recall = 0.0, 0.0

            if not precision + recall == 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            support = statistics[entity_type]["support"]
            clf_report[entity_type] = {"Precision": precision, "Recall": recall, "F1": f1, "Support": support}

        # Sort clf report descending
        clf_report = dict(sorted(clf_report.items(), key=lambda item: item[1]["Support"], reverse=True))

        # Compute micro F1 Scores
        tp_all = sum([statistics[entity_type]["tp"] for entity_type in self.entity_set])
        fp_all = sum([statistics[entity_type]["fp"] for entity_type in self.entity_set])
        fn_all = sum([statistics[entity_type]["fn"] for entity_type in self.entity_set])
        support_all = sum([statistics[entity_type]["support"] for entity_type in self.entity_set])

        if tp_all:
            micro_precision = 100 * tp_all / (tp_all + fp_all)
            micro_recall = 100 * tp_all / (tp_all + fn_all)
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        else:
            micro_precision, micro_recall, micro_f1 = 0.0, 0.0, 0.0

        clf_report["micro avg"] = {
            "Precision": micro_precision,
            "Recall": micro_recall,
            "F1": micro_f1,
            "Support": support_all,
        }

        # Compute Macro F1 Scores
        macro_precision = np.mean(
            [
                clf_report[entity_type]["Precision"]
                for entity_type in self.entity_set
                if clf_report[entity_type]["Support"] > 0
            ]
        )
        macro_recall = np.mean(
            [
                clf_report[entity_type]["Recall"]
                for entity_type in self.entity_set
                if clf_report[entity_type]["Support"] > 0
            ]
        )
        macro_f1 = np.mean(
            [clf_report[entity_type]["F1"] for entity_type in self.entity_set if clf_report[entity_type]["Support"] > 0]
        )

        clf_report["macro avg"] = {
            "Precision": macro_precision,
            "Recall": macro_recall,
            "F1": macro_f1,
            "Support": support_all,
        }

        return {"ner_clf_report": clf_report, "ner_micro_f1": micro_f1, "ner_macro_f1": macro_f1}

    def reset(self):
        self.ground_truth_entities = []
        self.entity_strings_predicted = []
        self.saved_observations = 0


class BLEU(Metric):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name)

        self.bleu = BLEUScore(**kwargs)

    def update(self, entity_string: List[str], entity_string_predicted: List[str]):
        self.bleu.update(preds=entity_string_predicted, target=entity_string)

    def reset(self):
        self.bleu.reset()

    def compute(self):
        return {"bleu": float(self.bleu.compute())}


METRICS = {"nerf1": NERF1, "bleu": BLEU}
