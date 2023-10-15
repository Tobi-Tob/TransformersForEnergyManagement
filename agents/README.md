# Add your agent here

Custom agents need to be a subclass of the Citylearn environment's Agent class. Specifically, they should implement the `predict` function that takes in a list of observations and returns the actions. Additionally, for AIcrowd evlauations you need to implement the `register_reset` function that will be called for the first set of observations only, this will help you do additional processing in the start of the episode if you want.

### Ideas to implement:
- Decompose the problem into separable subproblems:
  - Solar modul which predicts the next steps solar generation given the weather history predictions, current weather, current solar generation and building PV information (continuous learning?)
  - Heat pump controller which returns the action to control the heat pump. Input are the outdoor temp, the indoor temp difference, building specific values and maybe dependent on occupant count. The main module therefor only needs to control the storages.
  - Load modul which predicts the next steps electricity load depending on current values, time, occupant count
- Optimize Reward function (without tableaus to guide at every position towards the desired behaviour)
- Hyperparameter optimization
  - Network architecture, size and depth
  - Different activation functions
  - Algorithm specific hyperparameters
- Different feature representation (solar generation and non-shift able load normalize with yearly estimate)
- Artificially increase the rate of power outages to force the agent to learn this behavior, which is crucial to the challenge
- Other RL algorithms like SAC
- Data augmentation of buildings (linear combination or noise) for richer training
