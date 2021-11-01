import timeout_decorator
from loguru import logger

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, SimulatorConfig, EnvConfig


def run_evaluation():
    submission_config = SubmissionConfig()
    simulator_config = SimulatorConfig()
    env_config = EnvConfig()

    evaluator = Learn2RaceEvaluator(
        submission_config=submission_config,
        sim_config=simulator_config,
        env_config=env_config,
    )

    evaluator.create_env(["Thruxton"])
    evaluator.load_agent()

    try:
        evaluator.pre_evaluate()
    except timeout_decorator.TimeoutError:
        logger.info("Stopping pre-evaluation run")

    evaluator.evaluate()
    scores = evaluator.get_average_metrics()
    logger.success(f"Average metrics: {scores}")


if __name__ == "__main__":
    run_evaluation()
