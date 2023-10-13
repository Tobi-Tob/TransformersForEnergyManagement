from typing import Any, List

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv


class ZeroAgent(Agent):

    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.action_dim = self.action_space[0].shape[0]
        self.model_info = dict(
            model_id='ZeroAgent',
            action_dim=self.action_dim,
        )

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """ Just a passthrough, can implement any custom logic as needed """
        action = [[0] * self.action_dim]
        return action

    def set_model_index(self, idx):
        pass

    def next_model_index(self):
        pass
