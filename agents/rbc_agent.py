from citylearn.agents.rbc import BasicRBC

class BasicRBCAgent(BasicRBC):
    """ Can be any subclass of citylearn.agents.base.Agent """
    def __init__(self, env):
        super().__init__(env)

    def register_reset(self, observations):
        """ Register reset needs the first set of actions after reset """
        self.reset()
        return self.predict(observations)

    def predict(self, observations):
        """ Just a passthrough, can implement any custom logic as needed """
        return super().predict(observations) 