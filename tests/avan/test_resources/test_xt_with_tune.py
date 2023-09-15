from copy import deepcopy
import datetime
import subprocess
import time

import ray
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import os

from ray.tune.schedulers.pb2 import PB2
from ray.tune.examples.pbt_function import pbt_function
from ray.tune.trainable import Trainable

from create_yaml_tool import create_yaml_LR
from test_mo import get_throughput
from test_optuna_bys import find_max_th, Total_CPUs
from xt.main import main
from multiprocessing import Process, Lock, Semaphore
import optuna
# sem = Semaphore(4)
import os
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

lock1 = Lock()

def run_xt(yaml_file_path, run_log):
    os.system('xt_main -f ' + yaml_file_path)
    # main(yaml_file_path)
    # p = subprocess.Popen(
    #     '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + yaml_file_path + " " + run_log, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # for line in p.stdout.readlines():
    #     print(line)
    #     break
    print("good")


times = set()

# tune_system = False
# with_samper = False

tune_system = int(os.environ.get("tune_system", 1)) != 0
with_samper = int(os.environ.get("with_samper", 1)) != 0

print("tune_system ================== {}".format(tune_system))
print("with_samper ================== {}".format(with_samper))

perturbation_interval = 1

# def constraints(trial):
#     return trial.user_attrs["constraint"]


# sampler = optuna.integration.BoTorchSampler(
#     constraints_func=constraints,
#     n_startup_trials=1,
# )

def train_xt(config):
    print(config)
    # Create our data loaders, model, and optmizer.
    # global study
    # os.system("/bin/bash /home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test_resources/kill_xt.sh")
    # time.sleep(10)
    step = 1
    # train_loader, test_loader = get_data_loaders()
    # model = ConvNet()
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config.get("lr", 0.01),
    #     momentum=config.get("momentum", 0.9),
    # )

    start_run = time.time()

    raw_file_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_pipeline_128opt.yaml"

    update_dic = {}

    lr = config.get("lr", 0.01)
    update_dic.update({"LR": deepcopy(lr)})
    

    complete_step = 500000
    update_dic.update({"complete_step": complete_step})

    abs_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_test"
    # update_dic.update({"LR": config.get("lr", 0.01)})
    # yaml_file_path, bench_filepath = create_yaml_LR(raw_file_path, update_dic, abs_path)

   
    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.

    if session.get_checkpoint():
        
        # Load model state and iteration step from checkpoint.
        checkpoint_dict = session.get_checkpoint().to_dict()
        update_dic.update({"init_weights": checkpoint_dict["model_filepath"]})
        # print("=================== load model ======================")
        # print("checkpoint_dict[model_filepath]=================={}".format(checkpoint_dict["model_filepath"]))
        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

    
    while True:
        str_time = str(time.time()).replace(".", "")
        
        trial_id = session.get_trial_id()
        print("trail id ================= {}".format(trial_id))
        start_core = int(trial_id[-1]) * 10
        while str_time in times:
            time.sleep(0.01)
            str_time = str(time.time()).replace(".", "")
        times.add(str_time)
        update_dic.update({"now_time": str_time})
        update_dic.update({"start_core": start_core})
        yaml_file_path, bench_filepath = create_yaml_LR(raw_file_path, update_dic, abs_path)
        # print("yaml_file_path ================= {}".format(yaml_file_path))
        # print("bench_filepath ================= {}".format(bench_filepath))
        model_filepath = os.path.join(bench_filepath, "models/actor")
        # print("model_filepath ================= {}".format(model_filepath))

        # study = optuna.create_study(direction='maximize')
        n_trials = int(os.environ.get("trials", 4))

        if step >= int(os.environ.get("iter", 4)):
            n_trials = 0


        if tune_system:
            # print("=================tune_system=========================")
            # if with_samper:
            #     print("=================with_samper=========================")
            #     study = optuna.load_study(
            #         study_name="distributed-example", 
            #         storage="mysql://root:tanklab@localhost/example",
            #         sampler=sampler,
            #     )
            # else:
        
            #     study = optuna.load_study(
            #         study_name="distributed-example", 
            #         storage="mysql://root:tanklab@localhost/example"
            #     )
            print("tune with_samper ===================== {}".format(with_samper))
            best_param = find_max_th(yaml_file_path, n_trials, start_core, with_samper)
        
            update_dic.update({"size": best_param["env_pool_size"]})
            update_dic.update({"vector_env_size": best_param["env_pool_size"]})
            update_dic.update({"wait_nums": best_param["env_pool_wait_nums"]})
            infer_size = Total_CPUs - best_param["env_pool_size"]
            group_num = 1
            env_num = infer_size * group_num
            update_dic.update({"env_num": env_num})
            update_dic.update({"start_core": start_core})
            yaml_file_path, bench_filepath = create_yaml_LR(raw_file_path, update_dic, abs_path)
        
        run_log = "/tmp/test_xt_with_tune" + str_time + ".log"
        # print("best_param ================== {}".format(best_param))
        print("ray tune xt log  ================= {}".format(run_log))
        time.sleep(1)
        # p = subprocess.Popen(
        #     '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + yaml_file_path + " " + run_log, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # for line in p.stdout.readlines():
        #     print(line)
        #     break
        # os.system('xt_main -f ' + yaml_file_path)
        # main()
        # run_xt(yaml_file_path, run_log)
        p = Process(target=run_xt, args=(yaml_file_path, run_log,))
        p.start()
        pid = p.pid
        p.join()
        os.system("kill -9 " + str(pid))

        filename = "/tmp/"+str(str_time)
        throughput, reward = get_throughput([filename])
        print("tune ray throughput ============== {}".format(throughput))
        print("tune ray reward ============== {}".format(reward))

        time.sleep(1)
        checkpoint = None
        if step % config["checkpoint_interval"] == 0:
            # Every `checkpoint_interval` steps, checkpoint our current state.
            checkpoint = Checkpoint.from_dict({
                "step": step,
                "model_filepath": model_filepath,
            })

        session.report(
            {"mean_accuracy": reward, "lr": config["lr"]},
            checkpoint=checkpoint
        )
        step += 1
        end_run = time.time()
        print("ray run one round time ================== {}".format(end_run - start_run))

# scheduler = PopulationBasedTraining(
#     time_attr="training_iteration",
#     perturbation_interval=perturbation_interval,
#     metric="mean_accuracy",
#     mode="max",
#     hyperparam_mutations={
#         # distribution for resampling
#         "lr": tune.uniform(0.0001, 1),
#         # allow perturbations within this set of categorical values
#         "momentum": [0.8, 0.9, 0.99],
#     },
# )

scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric="mean_accuracy",
        mode="max",
        synch=True,
        hyperparam_bounds={
            # distribution for resampling
            "lr": [0.0001, 0.005],
            # allow perturbations within this set of categorical values
            # "momentum": (0.8, 0.99),
    },
)

if ray.is_initialized():
    ray.shutdown()
ray.init()

tuner = tune.Tuner(
    tune.with_resources(
        train_xt,
        resources={"cpu": 10, "gpu": 0.4}
    ),
    run_config=air.RunConfig(
        name="pb2_test4",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        # stop={"mean_accuracy": 300, "training_iteration": 20},
        stop={"training_iteration": 20},

        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=None,
        ),
        local_dir="/tmp/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space={
        "lr": tune.uniform(0.0001, 0.005),
        # "momentum": tune.uniform(0.001, 1),
        "checkpoint_interval": 1,
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