from typing import DefaultDict, List
from warnings import filterwarnings

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

filterwarnings(action="ignore", category=UserWarning)


def calculate_metrics(
    metrics: DefaultDict[str, List[float]],
    loss: float,
    y_true: List[List[int]],
    y_pred: List[List[int]],
) -> DefaultDict[str, List[float]]:
    """
    Calculate metrics on epoch.
    """

    smoothing_function = SmoothingFunction()
    bleu_score = corpus_bleu(
        list_of_references=[[seq] for seq in y_true],
        hypotheses=y_pred,
        smoothing_function=smoothing_function.method0,  # no smoothing
    )

    metrics["loss"].append(loss)
    metrics["bleu_score"].append(bleu_score)

    return metrics
