![Citylearn Banner](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1126/56b6e3e3143621e86382.png)

# [NeurIPS 2023 Citylearn Challenge](https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge) - Starter Kit 
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the NeurIPS 2023 Citylearn Challenge **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent, etc.
*  **Starter code** for you to get started!

# Table of Contents

- [Competition Overview](#competition-overview)
    + [Competition Phases](#competition-phases)
- [Getting Started](#getting-started)
- [How to write your own agent?](#how-to-write-your-own-agent)
- [How to start participating?](#how-to-start-participating)
  * [Setup](#setup)
  * [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime-dependencies)
  * [What should my code structure be like?](#what-should-my-code-structure-be-like)
  * [How to make a submission?](#how-to-make-a-submission)
- [Other Concepts](#other-concepts)
    + [Evaluation Metrics](#evaluation-metrics)
    + [Ranking Criteria](#ranking-criteria)
    + [Time constraints](#time-constraints)
  * [Local Evaluation](#local-evaluation)
  * [Contributing](#contributing)
  * [Contributors](#contributors)
- [Important links](#-important-links)


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
3. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-starter-kit/-/forks/new) to create a fork.
4. **Clone** your forked repo and start developing your agent.
5. **Develop** your agent(s) following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
5. **Develop** your reward function following the template in [how to write your own reward function](#how-to-write-your-own-reward-function) section.
6. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the citylearn simulator and report the metrics on the leaderboard of the competition.

# How to write your own agent?

We recommend that you place the code for all your agents in the `agents` directory (though it is not mandatory). You should implement the

- `register_reset` - The function is used to get the prediction of the first observation after `env.reset`
- `predict` - This function is called to get actions all observations, except the first.

**Add your agent name in** `user_agent.py`, this is what will be used for the evaluations.
  
An example are provided in `agents/rbc_agent.py`

# How to write your own reward function?

We recommend that you place the code for all your reward functions in the `rewards` directory (though it is not mandatory). You should implement the

- `calculate` - The function is used to get the reward for all observations.

**Add your reward class name in** `user_reward.py`, this is what will be used for the evaluations.
  
An example are provided in `rewards/comfort_reward.py`

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/citylearn-challenge/citylearn-2023-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:aicrowd/challenges/citylearn-challenge/citylearn-2023-starter-kit.git
    ```

3. **Install** competition specific dependencies!
    ```
    cd citylearn-2023-starter-kit
    pip install -r requirements.txt
    ```

4. Write your own agent as described in [How to write your own agent](#how-to-write-your-own-agent) section.

4. Write your own reward function as described in [How to write your own reward function](#how-to-write-your-own-reward-function) section.

5. Test your agent locally using `python local_evaluation.py`

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
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Linux packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Use this to check your agent evaluation flow locally
├── data/                  # Contains schema files for citylearn simulator (for local testing)
└── agents                 # Place your agents related code here
    ├── rbc_agent.py               # Simple rule based agent
    └── user_agent.py              # IMPORTANT: Add your agent name here
└── rewards                 # Place your reward related code here
    ├── comfort_reward.py          # Example reward function
    └── user_reward.py              # IMPORTANT: Add your reward class here
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!**

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "control-track-citylearn-challenge",
  "authors": ["your-aicrowd-username"],
  "description": "(optional) description about your awesome agent",
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above.

## How to make a submission?

👉 [submission.md](/docs/submission.md)

**Best of Luck** :tada: :tada:

# Other Concepts
### Evaluation Metrics

The [control track score](#eqn-control-track-score) (Score<sub>Control</sub>) is the weighted average of a thermal comfort score (Score<sub>Control</sub><sup>Comfort</sup>), an emissions score (Score<sub>Control</sub><sup>Emissions</sup>), a grid score (Score<sub>Control</sub><sup>Grid</sup>), and a resilience score (Score<sub>Control</sub><sup>Resilience</sup>). Score<sub>Control</sub><sup>Grid</sup> and Score<sub>Control</sub><sup>Resilience</sup> are averages of four and two KPIs respectively.

<a name="eqn-control-track-score"></a>
```math
\textrm{Score}_{\textrm{Control}} = w_1 \cdot \textrm{Score}_{\textrm{Control}}^{\textrm{Comfort}} + w_2 \cdot \textrm{Score}_{\textrm{Control}}^{\textrm{Emissions}} + w_3 \cdot \textrm{Score}_{\textrm{Control}}^{\textrm{Grid}} + w_4 \cdot \textrm{Score}_{\textrm{Control}}^{\textrm{Resilience}}
```

```math
\textrm{Score}_{\textrm{Control}}^{\textrm{Comfort}} = U
```
```math
\textrm{Score}_{\textrm{Control}}^{\textrm{Emissions}} = G
```

```math
\textrm{Score}_{\textrm{Control}}^{\textrm{Grid}} = \overline{R, L, P_d, P_n}
```

```math
\textrm{Score}_{\textrm{Control}}^{\textrm{Resilience}} = \overline{M, S}
```

The weights are specified in the [table below](#tab-control-track-weight). In Phase I, the highest weight is given to Score<sub>Control</sub><sup>Grid</sup>. By Phase II, Score<sub>Control</sub><sup>Resilience</sup> is introduced and has the same weight as Score<sub>Control</sub><sup>Comfort</sup> and Score<sub>Control</sub><sup>Grid</sup>. Score<sub>Control</sub><sup>Emissions</sup> has the lowest non-zero weight in both phases. The private leaderboard in Phase II makes use of the same weights as the public leaderboard.

<a name="tab-control-track-weight"></a>
| Phase | $`w_1`$ | $`w_2`$ | $`w_3`$ | $`w_4`$ |
|-|-|-|-|-|
Phase I | 0.3 | 0.1 | 0.6 | 0.0 |
Phase II | 0.3 | 0.1 | 0.3 | 0.3 |


All together, the four scores are made up of eight KPIs namely: carbon emissions (_G_), discomfort (_U_), ramping (_R_), 1 - load factor (_L_), daily peak (_P\_d_), all-time peak (_P\_n_) 1 - thermal resilience (_M_), and normalized unserved energy (_S_), which have been defined in the [table below](#tab-control-track-score). _G_, _U_, _M_, and _S_ are building-level KPIs that are calculated using each building's net electricity consumption (_e_) or temperature (_T_) then averaged to get the neighborhood-level value. _R_, _L_, _P\_d_, and _P\_n_ are neighborhood-level KPIs that are calculated using the neighborhood's net electricity consumption (_E_). Except _U_, _M_, and _S_ all KPIs are normalized by their baseline value where the baseline is the result from when none of the distributed energy resources (DHW storage system, battery, and heat pump) is controlled.

<a name="tab-control-track-score"></a>
| Name | Formula | Description |
|-|-|-|
Carbon emissions | $`G = \sum_{i=0}^{b-1} g^i_{\textrm{control}} \div \sum_{i=0}^{b-1} g^i_{\textrm{baseline}} \\ g = \sum_{t=0}^{n-1}{\textrm{max} \left (0,e_t \cdot B_t \right )}`$ | Emissions from imported electricity. | <!--END-->
Discomfort | $`U = \sum_{i=0}^{b-1} u^i_{\textrm{control}} \div b \\ u = a \div o \\ a = \sum_{t=0}^{n-1} \begin{cases} 1, \ \textrm{if} \ \lvert T_t - T_t^{\textrm{setpoint}} \rvert > c \ \textrm{and} \ O_t > 0 \\ 0  \end{cases} \\ o = \sum_{t=0}^{n-1} \begin{cases} 1, \ \textrm{if} \ O_t > 0 \\ 0 \end{cases}`$ | Proportion of time steps when a building is occupied, and indoor temperature falls outside a comfort band, $`c`$. | <!--END-->
Ramping | $`R = r_{\textrm{control}} \div r_{\textrm{baseline}} \\ r = \sum_{t=0}^{n-1}  \lvert E_{t} - E_{t - 1} \rvert`$ | Smoothness of the neighborhood’s consumption profile where low $`R`$ means there is gradual increase in consumption even after self-generation is unavailable in the evening and early morning. High $`R`$ means abrupt change in grid load that may lead to unscheduled strain on grid infrastructure and blackouts caused by supply deficit. | <!--END-->
1 - Load factor | $`L = l_{\textrm{control}} \div l_{\textrm{baseline}} \\ l = \Bigg ( \sum_{d=0}^{n \div h} 1 - \frac{ \left ( \sum_{t=d \cdot h}^{d \cdot h +  h - 1} E_{t} \right ) \div h }{ \textrm{max} \left (E_{d \cdot h}, \dots, E_{d \cdot h +  h - 1} \right ) } \Bigg ) \div \Bigg ( \frac{n}{h} \Bigg)`$ | Average ratio of daily average and peak consumption. Load factor is the efficiency of electricity consumption and is bounded between 0 (very inefficient) and 1 (highly efficient) thus, the goal is to maximize the load factor or minimize (1 − load factor)| <!--END-->
Daily peak | $`P_d = p_{d_{\textrm{control}}} \div p_{d_{\textrm{baseline}}} \\ p_d = \Bigg ( \sum_{d=0}^{n \div h} \textrm{max} \left (E_{d \cdot h}, \dots, E_{d \cdot h +  h - 1} \right )\Bigg ) \div \Bigg ( \frac{n}{h} \Bigg)`$ | Average, maximum consumption at any time step per day. | <!--END-->
All-time peak | $`P_n = p_{n_{\textrm{control}}} \div p_{n_{\textrm{baseline}}} \\ p_n = \textrm{max} \left (E_{0}, \dots, E_{n} \right )`$ | Maximum consumption at any time step. | <!--END-->
1 - Thermal resilience | $`M = \sum_{i=0}^{b-1} m^i_{\textrm{control}} \div b \\ m = a \div o \\ a = \sum_{t=0}^{n-1} \begin{cases} 1, \ \textrm{if} \ \lvert T_t - T_t^{\textrm{setpoint}} \rvert > c \ \textrm{and} \ O_t > 0 \ \textrm{and} \ F_t > 0 \\ 0  \end{cases} \\ o = \sum_{t=0}^{n-1} \begin{cases} 1, \ \textrm{if} \ O_t > 0 \ \textrm{and} \ F_t > 0 \\ 0 \end{cases}`$ | Same as discomfort, $`U`$ but only considers time steps when there is power outage. | <!--END-->
Normalized unserved energy | $`S = \sum_{i=0}^{b-1} s^i_{\textrm{control}} \div b \\ s = s^{\textrm{served}} \div s^{\textrm{expected}} \\ s^{\textrm{served}} = \sum_{t=0}^{n-1} \begin{cases} q_{n}^{\textrm{served}}, \ \textrm{if} \ F_t > 0 \\ 0 \end{cases} \\ s^{\textrm{expected}} = \sum_{t=0}^{n-1} \begin{cases} q_{n}^{\textrm{expected}}, \ \textrm{if} \ F_t > 0 \\ 0 \end{cases}`$ | Proportion of unmet demand due to supply shortage e.g. power outage. | <!--END-->

> Where:
> - $`t`$: Time step index;
> - $`n`$: Total number of time steps, $`t`$, in 1 episode;
> - $`h`$: Hours per day (24);
> - $`d`$: Day;
> - $`i`$: Building index;
> - $`b`$: Total number of buildings;
> - $`e`$: Building-level net electricity consumption (kWh);
> - $`E`$: Neighborhood-level net electricity consumption (kWh);
> - $`A`$: Electricity rate ($/kWh);
> - $`B`$: Emission rate (kg<sub>CO<sub>2</sub>e</sub>/kWh);
> - $`T`$: Indoor dry-bulb temperature (<sup>o</sup>C);
> - $`T^{\textrm{setpoint}}`$: Indoor dry-bulb temperature setpoint (<sup>o</sup>C);
> - $`c`$: Thermal comfort band ($`\pm T^{\textrm{setpoint}}`$);
> - $`O`$: Occupant count (people);
> - $`F`$: Power outage signal (Yes/No); and
> - $`q`$: Building-level cooling, domestic hot water and non-shiftable load energy demand (kWh).
### Time and compute constraints

For Phase I, your agent should complete `1 episode in 30 minutes`. Note that the number of episodes and time can change depending on the phase of the challenge. However we will try to keep the throughput requirement of your agent, so you need not worry about phase changes. We only measure the time taken by your agent. For compute, you will be provided a virutal machine with `2 CPU cores and 12 GB of RAM`.

## Local Evaluation

Participants can run the evaluation protocol for their agent locally with or without any constraint posed by the Challenge to benchmark their agents privately. See `local_evaluation.py` for details. You can change it as you like, it will not be used for the competition. You can also change the simulator schema provided under `data/schemas/warm_up/schema.json`, this will not be used for the competition.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Import it in `user_agent.py`
- Test it out using `python local_evaluation.py`.
- Add any documentation for your approach at top of your file.
- Create merge request! 🎉🎉🎉 

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge

- 🗣 Discussion Forum: https://discourse.aicrowd.com/c/neurips-2023-citylearn-challenge

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/neurips-2023-citylearn-challenge/problems/control-track-citylearn-challenge/leaderboards
