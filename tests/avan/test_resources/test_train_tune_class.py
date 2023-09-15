from ray import air, tune

def objective(x, a, b):
    return a * (x ** 0.5) + b

class Trainable(tune.Trainable):
    def setup(self, config: dict):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def step(self):  # This is called iteratively.
        print("self.trial_id ==================== {}".format(self.trial_id))
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {"score": score}


tuner = tune.Tuner(
    Trainable,
    run_config=air.RunConfig(
        # Train for 20 steps
        stop={"training_iteration": 20},
        checkpoint_config=air.CheckpointConfig(
            # We haven't implemented checkpointing yet. See below!
            checkpoint_at_end=False
        ),
    ),
    param_space={"a": 2, "b": 4},
)
results = tuner.fit()