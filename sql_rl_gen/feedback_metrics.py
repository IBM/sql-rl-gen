from typing import List, Tuple, Dict
from collections import Counter

def calculate_all_feedback_metrics(expected_response: List[Tuple], response_from_generated_query: List[Tuple]) -> Dict[str, float]:
    try:
        expected_set = Counter(expected_response)
        response_set = Counter(response_from_generated_query)
        intersection = expected_set & response_set
        union = expected_set | response_set
        true_positives = sum(intersection.values())
        false_positives = sum((response_set - expected_set).values())
        false_negatives = sum((expected_set - response_set).values())
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = true_positives / sum(union.values()) if sum(union.values()) > 0 else 0.0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 1
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou, "not_sql_format": False, "forbidden_sql_command": False, "error_type": None, "error_reason": None}
    except Exception as e:
        return {"accuracy": -1, "precision": -1, "recall": -1, "f1": -1, "iou": -1, "not_sql_format": False, "forbidden_sql_command": False, "error_type": type(e), "error_reason": str(e)}