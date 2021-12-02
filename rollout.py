import tempfile
import timeout_decorator
from loguru import logger
from ruamel.yaml import YAML
import os, re

from evaluator.evaluator import Learn2RaceEvaluator
from config import SubmissionConfig, SimulatorConfig, EnvConfig
from common.utils import find_envvar_patterns, replace_envvar_patterns, resolve_envvars

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
    simulator_config = SimulatorConfig()
    env_config = EnvConfig()
    
    yaml = YAML()
    params = yaml.load(open('params-sac.yaml'))
    sac_kwargs = resolve_envvars(params['sac_kwargs'], args())

    evaluator = Learn2RaceEvaluator(
        submission_config=submission_config,
        sim_config=simulator_config,
        env_config=env_config,
        sac_config = sac_kwargs
    )

    evaluator.create_env(["Thruxton"])
    evaluator.init_agent()

    ## Local development OR Stage 2 'practice' phase
    training_routine(evaluator)
    ## Evaluation
    evaluation_routine(evaluator)


if __name__ == "__main__":
    run_evaluation()
