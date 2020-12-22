from typing import DefaultDict, List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def calculate_metrics(
    metrics: DefaultDict[str, List[float]],
    loss: float,
    y_true: List[int],
    y_pred: List[int],
) -> DefaultDict[str, List[float]]:
    """
    Calculate metrics on epoch.
    """

    smoothing_function = SmoothingFunction()
    bleu = sentence_bleu(
        references=[y_true],
        hypothesis=y_pred,
        smoothing_function=smoothing_function.method4,
    )

    metrics["loss"].append(loss)
    metrics["bleu"].append(bleu)

    return metrics