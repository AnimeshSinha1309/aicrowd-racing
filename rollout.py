import logging
import tempfile
import timeout_decorator

import wandb

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, EnvConfig, SimulatorConfig


pre_evaluate_model_file, pre_evaluate_model_path = tempfile.mkstemp()


def training_routine(evaluator):
    try:
        evaluator.train()
    except timeout_decorator.TimeoutError:
        logging.info("Stopping pre-evaluation run")

    evaluator.save_agent_model(pre_evaluate_model_path)


def evaluation_routine(evaluator):
    evaluator.load_agent_model(pre_evaluate_model_path)
    scores = evaluator.evaluate()
    logging.info(f"Success: Average metrics: {scores}")


def run_evaluation():
    evaluator = Learn2RaceEvaluator(
        submission_config=SubmissionConfig,
        sim_config=SimulatorConfig,
        env_config=EnvConfig,
    )

    evaluator.create_env()
    evaluator.init_agent()

    # Local development OR Stage 2 'practice' phase
    training_routine(evaluator)
    # Evaluation
    evaluation_routine(evaluator)


if __name__ == "__main__":
    wandb.init(project="aicrowd-racing", name="sac-training-1", save_code=False)

    run_evaluation()
