from typing import Tuple, Dict
from textrl import TextRLEnv
from sql_rl_gen.generation.envs.utils import sql_query_execution_feedback_on_dataset, save_dict_csv

KEY_WORDS = ["SELECT", "FROM", "WHERE", "JOIN", "INNER", "OUTER", "LEFT", "RIGHT", "AS", "ON", "EXCEPT", "DISTINCT", "GROUP BY", "ORDER BY", "NOT", "ASC", "DESC", "LIMIT", "LIKE", "COUNT", "SUM", "AVG", "MIN", "MAX"]

def missed_key_words(base_penalty: float, input: str, predicted: str):
    keyword_counts_input = {}
    for keyword in KEY_WORDS:
        if keyword in input.upper() and keyword not in keyword_counts_input:
            keyword_counts_input[keyword] = input.upper().count(keyword)
    keyword_counts_predicted = {}
    for keyword in KEY_WORDS:
        if keyword in predicted.upper() and keyword not in keyword_counts_predicted:
            keyword_counts_predicted[keyword] = predicted.upper().count(keyword)
    total_keywords_input = sum(keyword_counts_input.values())
    num_missing_keywords = 0
    for keyword, count in keyword_counts_input.items():
        if keyword not in keyword_counts_predicted or count > keyword_counts_predicted.get(keyword, 0):
            num_missing_keywords += count - keyword_counts_predicted.get(keyword, 0)
    if len(keyword_counts_predicted) > len(keyword_counts_input):
        for keyword, count in keyword_counts_predicted.items():
            if keyword not in keyword_counts_input or count > keyword_counts_input.get(keyword, 0):
                num_missing_keywords += count - keyword_counts_input.get(keyword, 0)
    penalty = base_penalty - abs(num_missing_keywords) if total_keywords_input > 0 else -10.0
    return penalty

class SQLRLEnv(TextRLEnv):
    def __init__(self, model, tokenizer, dataset, dataset_path, output_dir, logger, environment_name, columns_names_mismatch=None, observation_input=[], max_length=1000, compare_sample=2,
                 unfreeze_layer_from_past=0):
        super().__init__(model, tokenizer, observation_input, max_length, compare_sample, unfreeze_layer_from_past)
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.environment_name = environment_name
        self.logger = logger
        self.columns_names_mismatch = columns_names_mismatch

    def render(self, mode="human"):
        pass

    def sql_query_execution_feedback(self, input_item, predicted_text) -> Dict:
        return sql_query_execution_feedback_on_dataset(self.dataset, self.dataset_path, input_item['input'], predicted_text, self.columns_names_mismatch)

    # Skeleton of the reward function
    def get_reward(self, input_item, predicted_list, finish):
        if finish:
            for predicted_value in predicted_list:
                if predicted_value[-1] == '</s>':
                    predicted_list[predicted_list.index(predicted_value)] = predicted_value[:-1]
            predicted_text = self.tokenizer.convert_tokens_to_string(predicted_list[0])
            reward, metrics = self.compute_reward(input_item, predicted_text)
            self.logger.info(f"Reward: {reward}, Metrics: {metrics}")
            metrics["reward"] = reward
            metrics_keys_sorted = list(metrics.keys())
            metrics_keys_sorted.sort()
            metrics_sorted = {i: metrics[i] for i in metrics_keys_sorted}
            save_dict_csv(metrics_sorted, self.output_dir, f"{self.environment_name}_metrics.csv")
            return reward
        return 0.0

    # Skeleton of the reward function
    def compute_reward(self, input_item, predicted_text) -> Tuple[float, Dict]