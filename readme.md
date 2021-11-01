// todo: replace banner

![Learn to Race Banner]()

# [ICLR 2021 - Learn to Race Challenge](https://www.aicrowd.com/challenges/iclr-2021-learn-to-race/) | Starter Kit 
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the ICLR 2021 Learn to Race Challenge **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

//todo: Add starter notebook

> **NOTE:** 
If you are resource-constrained or would not like to setup everything in your system, you can make your submission from inside Google Colab too. [**Check out the beta version of the Notebook.**](https://colab.research.google.com/drive/14FpktUXysnjIL165hU3rTUKPHo4-YRPh?usp=sharing)



# Table of Contents

1. [Competition Procedure](#competition-procedure)
2. [How to access and use dataset](#how-to-access-and-use-dataset)
3. [How to start participating](#how-to-start-participating)
4. [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies-)
5. [What should my code structure be like ?](#what-should-my-code-structure-be-like-)
6. [How to make submission](#how-to-make-submission)
7. [Other concepts](#other-concepts)
8. [Important links](#-important-links)


#  Competition Procedure

The ICLR 2021 Learn to Race Challenge is an opportunity for researchers and machine learning enthusiasts to test their skills by creating a system able to ...

In this challenge, you will train your models locally and then upload them to AIcrowd (via git) to be evaluated. 

**The following is a high level description of how this process works**

// todo: replace image with competition specific one

![](https://i.imgur.com/xzQkwKV.jpg)

1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/iclr-2021-learn-to-race/).
2. **Clone** this repo and start developing your solution.
3. **Train** your models for ... and write your agent as described in [how to write your own agent](#how-to-write-your-own-agent) section.
4. [**Submit**](#how-to-submit-a-model) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-submit-a-model). The automated evaluation setup will evaluate the submissions against the test dataset to compute and report the metrics on the leaderboard of the competition.

# How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). Any agent you write should implement the `compute_action` and `pre_evaluate` methods as follows.

```python
from agents.base import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self):
        # Do something here
        pass

    def compute_action(self, state):
        # Do something here
        return 1  # return some action

    def pre_evaluate(self, env):
        # Do something here
        # You are responsible to make sure that this function doesn't take
        # beyond 1 hour. Anything beyond 1 hour will be forcefully stopped
        # leading to bad state.
        pass
```

Update the `SubmissionConfig` in [config.py](config.py#L5) to use your new agent class instead of the `RandomAgent`.

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:learn-to-race/learn-to-race-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd learn-to-race-starter-kit
    pip install -r requirements.txt
    ```

4. Try out the random agent behaviour by running `python rollout.py`.

5. Write your own agent as described in [how to write your own agent](#how-to-write-your-own-agent) section.

6. Make a submission as described in [how to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies ?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `environment.yml` (conda environment), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [RUNTIME.md](docs/runtime.md) file.

## What should my code structure be like ?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── rollout.py             # Entrypoint to test your code locally (DO NOT EDIT, will be replaced during evaluation)
├── config.py              # File containing env, simulator and submission configuration
└── agents                 # Place your agents related code here
    ├── base.py            # Code for base agent
    └── <my_agent>.py      # IMPORTANT: Your agent code
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!** 

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "evaluations-api-music-demixing",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
  "external_dataset_used": false
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make a submission?

👉 [SUBMISSION.md](/docs/SUBMISSION.md)

**Best of Luck** :tada: :tada:

# Other Concepts

## Time constraints

// todo: Add time constraints (+ timestep budget, max episodes, etc.,)

## Local Evaluation

// todo: How to run evaluations locally? (ideally just run `python rollout.py` after setup)

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Test it out using `python rollout.py`.
- Add any documentation for your approach at top of your file.
- Import it in `config.py`
- Create merge request! 🎉🎉🎉 

## Contributors

// todo

# 📎 Important links


💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/iclr-2021-learn-to-race

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/iclr-2021-learn-to-race/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/iclr-2021-learn-to-race/leaderboards
