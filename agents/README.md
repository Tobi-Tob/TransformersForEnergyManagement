Custom agents need to be a subclass of the Citylearn environment's Agent class.
Specifically, they should implement the `predict` function that takes in a list of observations and returns the actions.
Additionally, for AIcrowd evlauations you need to implement the `register_reset` function that will be called for the
first set of observations only, this will help you do additional processing in the start of the episode if you want.
