from dataclasses import dataclass, field
from enum import Enum

@dataclass
class GeneralTrainingArguments:
    outdir: str = field(
        metadata={
            "help": "The directory in which to save the trained agent"
        }
    )
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models."
        }
    )
    number_of_rows_to_use: int = field(
        default=1000,
        metadata={
            "help": "Number of rows to use from dataset."
        }
    )

@dataclass
class MandatoryTrainingArguments(GeneralTrainingArguments):
    temperature: float = field(
        default=0.8,
        metadata={
            "help": "Temperature parameter for model training."
        }
    )
    top_k: int = field(
        default=100,
        metadata={
            "help": "Top k predictions to show."
        }
    )
    top_p: float = field(
        default=0.85,
        metadata={
            "help": "Top p predictions to show."
        }
    )
    update_interval: int = field(
        default=50,
        metadata={
            "help": "Update interval in episodes."
        }
    )
    minibatch_size: int = field(
        default=512,
        metadata={
            "help": "Minibatch size for training."
        }
    )
    epochs: int = field(
        default=5000,
        metadata={
            "help": "Total number of epochs to train."
        }
    )
    lr: float = field(
        default=3e-4,
        metadata={
            "help": "Learning rate."
        }
    )

@dataclass
class TrainingArguments(MandatoryTrainingArguments):
    steps_n: int = field(
        default=1000,
        metadata={
            "help": "Total number of steps to train."
        }
    )
    eval_n_episodes: int = field(
        default=5,
        metadata={
            "help": "Number of episodes to evaluate the agent on."
        }
    )
    train_max_episode_len: int = field(
        default=1000,
        metadata={
            "help": "Max episode length for training."
        }
    )
    eval_interval: int = field(
        default=10,
        metadata={
            "help": "Evaluation interval in episodes."
        }
    )

class EvaluationMethod(Enum):
    KFOLD = "kfold"
    TEST = "test"

@dataclass
class EvaluateArguments(MandatoryTrainingArguments):
    trained_agent_path: str = field(
        default=None,
        metadata={
            "help": "The trained agent to evaluate."
        }
    )
    evaluation_method: EvaluationMethod = field(
        default=EvaluationMethod.TEST,
        metadata={
            "help": "Either kfold or test",
        }
    )
