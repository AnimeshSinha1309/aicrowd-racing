import tempfile
import timeout_decorator
from loguru import logger
from ruamel.yaml import YAML
import os, re

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, SimulatorConfig, EnvConfig
from common.utils import resolve_envvars

class args:
    yaml = 'params-sac.yaml'
    dirhash = ''
    runtime = 'local'

pre_evaluate_model_file, pre_evaluate_model_path = tempfile.mkstemp()

def training_routine(evaluator):
    try:
        evaluator.train()
    except timeout_decorator.TimeoutError:
        logger.info("Stopping pre-evaluation run")

    evaluator.save_agent_model(pre_evaluate_model_path)

def evaluation_routine(evaluator):
    evaluator.load_agent_model(pre_evaluate_model_path)
    scores = evaluator.evaluate()
    logger.success(f"Average metrics: {scores}")

def run_evaluation():
    submission_config = SubmissionConfig()
    yaml = YAML()
    sys_params = yaml.load(open("configs/params-env.yaml"))
    env_kwargs = resolve_envvars(sys_params['env_kwargs'], args)
    sim_kwargs = resolve_envvars(sys_params['sim_kwargs'], args)

    evaluator = Learn2RaceEvaluator(
        submission_config=submission_config,
        sim_config=sim_kwargs,
        env_config=env_kwargs
    )

    evaluator.create_env()
    evaluator.init_agent()

    ## Local development OR Stage 2 'practice' phase
    training_routine(evaluator)
    ## Evaluation
    evaluation_routine(evaluator)


if __name__ == "__main__":
    run_evaluation()
