from multiprocessing import Process
import torch
import torch.optim as optim

import ray
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, train, test
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
import warnings
from ray.tune.trainable import Trainable

warnings.filterwarnings("ignore",category=DeprecationWarning)

def train_dl(model, optimizer, train_loader, test_loader):
    train(model, optimizer, train_loader)
    acc = test(model, test_loader)
    return acc

class TestTrainable(Trainable):
    def step(self):
        result = {"name": self.trial_name, "trial_id": self.trial_id}
        print(result)
        return result

class TestTrainable(tune.Trainable):
    def setup(self, config: dict):
        # config (dict): A dict of hyperparameters
        self.config = config

    def step(self):  # This is called iteratively.
        score = train_convnet(self.config)
        # self.x += 1
        # return {"score": score}


def train_convnet(config):
    # Create our data loaders, model, and optmizer.
    step = 1
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
    )
    print(type(config.get("lr", 0.01)))
    trial_id = session.get_trial_id()
    print("trail id ================= {}".format(trial_id))
    start_core = int(trial_id[-1]) * 10
    print("start_core ================= {}".format(start_core))

    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.
    if session.get_checkpoint():
        # Load model state and iteration step from checkpoint.
        checkpoint_dict = session.get_checkpoint().to_dict()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        # Load optimizer state (needed since we're using momentum),
        # then set the `lr` and `momentum` according to the config.
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "momentum" in config:
                param_group["momentum"] = config["momentum"]

        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1
        print("load checkpoint ========================== {}".format(step))
    print("before train ====================== {}".format(step))

    while True:
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        # p = Process(target=train_dl, args=(model, optimizer, train_loader, test_loader,))
        # p.start()
        # p.join()
        # acc = 0.5
        checkpoint = None
        # print("step ========================== {}".format(step))
        if step % config["checkpoint_interval"] == 0:
            # Every `checkpoint_interval` steps, checkpoint our current state.
            checkpoint = Checkpoint.from_dict({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            })

        session.report(
            {"mean_accuracy": acc, "lr": config["lr"]},
            checkpoint=checkpoint
        )
        step += 1
        print("after train ====================== {}".format(step))

perturbation_interval = 1
scheduler = PB2(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="mean_accuracy",
    mode="max",
    synch=True,
    hyperparam_bounds={
        # distribution for resampling
        "lr": [0.0001, 1],
        # allow perturbations within this set of categorical values
        # "momentum": [0.8, 0.9, 0.99],
        "momentum": [0.8, 0.99],
    },
)

if ray.is_initialized():
    ray.shutdown()
ray.init()

tuner = tune.Tuner(
    train_convnet,
    run_config=air.RunConfig(
        name="pbt_test",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        stop={"mean_accuracy": 0.96, "training_iteration": 50},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=4,
        ),
        local_dir="/tmp/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space={
        "lr": tune.uniform(0.001, 1),
        "momentum": tune.uniform(0.001, 1),
        "checkpoint_interval": perturbation_interval,
    },
)

results_grid = tuner.fit()


import matplotlib.pyplot as plt
import os

# Get the best trial result
best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")

# Print `log_dir` where checkpoints are stored
print('Best result logdir:', best_result.log_dir)

# Print the best trial `config` reported at the last iteration
# NOTE: This config is just what the trial ended up with at the last iteration.
# See the next section for replaying the entire history of configs.
print('Best final iteration hyperparameter config:\n', best_result.config)

# Plot the learning curve for the best trial
df = best_result.metrics_dataframe
# Deduplicate, since PBT might introduce duplicate data
df = df.drop_duplicates(subset="training_iteration", keep="last")
df.plot("training_iteration", "mean_accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Test Accuracy")
plt.show()