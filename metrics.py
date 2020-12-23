from typing import DefaultDict, List, Optional
from warnings import filterwarnings

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

filterwarnings(action="ignore", category=UserWarning)


def calculate_metrics(
    metrics: DefaultDict[str, List[float]],
    loss: float,
    y_true: List[List[int]],
    y_pred: List[List[int]],
    grad_norm: Optional[float] = None,
) -> DefaultDict[str, List[float]]:
    """
    Calculate metrics on epoch.
    """

    smoothing_function = SmoothingFunction()
    bleu_score = 100 * corpus_bleu(  # from 0 to 100
        list_of_references=[[seq] for seq in y_true],
        hypotheses=y_pred,
        smoothing_function=smoothing_function.method0,  # no smoothing
    )

    metrics["loss"].append(loss)
    metrics["bleu_score"].append(bleu_score)

    if grad_norm is not None:
        metrics["grad_norm"].append(grad_norm)

    return metrics
