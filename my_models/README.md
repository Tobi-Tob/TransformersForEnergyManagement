![Citylearn Banner](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1127/ce825c2f02b99c6fdb09.png)

# Add your forecast model here

Your model needs to be a subclass of the `base_predictor_model` class. Specifically, it should implement the `compute_forecast` function that takes in a list of observations and returns the forecasted results.

Refer to the model in (`example_predictor.py`) example and create your models in the same format


# Table of Contents

- [NeurIPS 2023 Citylearn Challenge Forecasting Track - Starter Kit](#neurips-2023-citylearn-challenge-forecasting-track---starter-kit)
- [Table of Contents](#table-of-contents)
- [Competition Overview](#competition-overview)
    - [Competition Phases](#competition-phases)
- [Getting Started](#getting-started)
- [How to write your own model?](#how-to-write-your-own-model)
- [How to start participating?](#how-to-start-participating)
  - [Setup](#setup)
  - [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime--dependencies)
  - [What should my code structure be like?](#what-should-my-code-structure-be-like)
  - [How to make a submission?](#how-to-make-a-submission)
- [Other Concepts](#other-concepts)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Time and compute constraints](#time-and-compute-constraints)
  - [Local Evaluation](#local-evaluation)
  - [Contributing](#contributing)
- [📎 Important links](#-important-links)


#  Competition Overview
[The CityLearn Challenge 2023](https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge) 

Buildings are responsible for 30% of greenhouse gas emissions. At the same time __buildings are taking a more active role in the power system__ by providing benefits to the electrical grid. As such, buildings are an unexplored opportunity to __address climate change__.

The CityLearn Challenge makes use of the __CityLearn Gym environment__ as an opportunity to compete in investigating the potential of artificial intelligence (AI) and distributed control systems to tackle multiple problems within the built-environment domain. It is designed to attract a multidisciplinary participation audience including researchers, industry experts, sustainability enthusiasts and AI hobbyists as a means of crowd-sourcing solutions to these problems.

The CityLearn Challenge 2023 is a two-track challenge where either track is independent of, but may inform design choices in the other track. Both tracks make use of the same dataset of a synthetic single-family neighborhood and are run in parallel.

### Competition Phases

## <a name="phase-1"></a>Phase I (Warm Up Round, Aug 21 - Sep 18, 2023)
This phase provides participants with an opportunity to familiarize themselves with the competition, CityLearn environment and raise issues bordering on the problem statement, source code, dataset quality and documentation to be addressed by the organizers. A solution example will also be provided so that participants can test the submission process and see their submissions show up on the leaderboard. The submissions and leaderboard in this phase are not taken into account during [Phase II](#phase-2) and subsequent selection of winners in [Phase III](#phase-3).
    
## <a name="phase-2"></a>Phase II (Evaluation Round, Sep 19 - Oct 31, 2023)
This is the competition round. Participants will be able to see how each of their submissions rank against each other and how their latest submission ranks against other participants’ submissions in a public leaderboard. There are also changes made to the environment and online evaluation dataset in this phase. At the end of this phase, new submissions will be halted and existing submissions are evaluated against another different dataset but the scores and rankings are kept private and visible to only the challenge organizers.
    
## <a name="phase-3"></a>Phase III (Review Round, Nov 1 - Nov 15, 2023)
During this phase, winners will be selected and announced. Also, the organizers will develop an executive summary of the competition that documents the preparation, winning solutions, challenges faced and lessons learned.

#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge).
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-forecasting-track-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your model.
5. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the citylearn simulator and report the metrics on the leaderboard of the competition.

# How to write your own model?

We recommend that you place the code for all your models in the `models` directory (though it is not mandatory). You should implement the

* `compute_forecast` - This function to compute the forecasted future state of the environment

**Add your model name in** `user_model.py`, this is what will be used for the evaluations.
  
An example are provided in `my_models/example_predictor.py`

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-forecasting-track-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:aicrowd/challenges/citylearn-challenge/citylearn-2023-forecasting-track-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd citylearn-2023-forecasting-track-starter-kit
    pip install -r requirements.txt
    ```

4. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

5. Test your model locally using `python local_forecast_evaluation.py`

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

You can check detailed information about the same in the 👉 [runtime.md](docs/runtime.md) file.

## What should my code structure be like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json                   # Submission meta information - like your username
├── apt.txt                        # Linux packages to be installed inside docker image
├── requirements.txt               # Python packages to be installed
├── local_forecast_evaluation.py   # Use this to check your model evaluation flow locally
├── data/                          # Contains schema files for citylearn simulator (for local testing)
└── my_models                      # Place your models related code here
    ├── example_predictor.py       # Simple example predictor
    └── user_model.py              # IMPORTANT: Add your model name here
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!**

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "forecasting-track-citylearn-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome model"
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make a submission?

👉 [submission.md](/docs/submission.md)

**Best of Luck** :tada: :tada:

# Other Concepts
### Evaluation Metrics

The [forecast track score](#eqn-forecast-track-score), Score<sub>Forecast</sub>, is the average over all of the variables being forecast, of the normalized mean root mean square error (RMSE) of the forecasts made.

<a name="eqn-forecast-track-score"></a>
```math
\textrm{Score}_{\textrm{Forecast}} = \frac{1}{V} \sum_v \left( \frac{\sum_{t=0}^{n-1} \sqrt{ \frac{1}{w} \sum_{\tau=1}^{w} \left(f^v_{t,\tau} - v_{t+\tau} \right)^2 } }{\sum_{t=0}^{n-1} v_t} \right)
```

> Where:
> - $`t`$: Environment time step index;
> - $`n`$: Total number of time steps, $`t`$, in 1 episode;
> - $`\tau`$: Forecasting window time step index;
> - $`w`$: Length of forecasting window (48hrs);
> - $`b`$: Total number of buildings;
> - $`v`$: Forecasting variable;
> - $`V`$: Total number of variables to forecast ($`3b+2`$);
> - $`f^v_{t,\tau}`$: Forecast of variable $`v`$ for time step $`t+\tau`$, made at time $`t`$;

### Time and compute constraints

For Phase I, your model should complete `1 episode in 30 minutes`. Note that the number of episodes and time can change depending on the phase of the challenge. However we will try to keep the throughput requirement of your model, so you need not worry about phase changes. We only measure the time taken by your model. For compute, you will be provided a virutal machine with `2 CPU cores and 12 GB of RAM`.

## Local Evaluation

Participants can run the evaluation protocol for their models locally with or without any constraint posed by the Challenge to benchmark their ,models privately. See `local_evaluation.py` for details. You can change it as you like, it will not be used for the competition. You can also change the simulator schema provided under `data/schemas/warm_up/schema.json`, this will not be used for the competition.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `my_models/<your_model>.py`.
- Import it in `user_model.py`
- Test it out using `python local_forecast_evaluation.py`.
- Add any documentation for your approach at top of your file.
- Create merge request! 🎉🎉🎉 

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/forecasting-track-citylearn-challenge

- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/forecasting-track-citylearn-challenge/discussion

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/forecasting-track-citylearn-challenge/leaderboards
