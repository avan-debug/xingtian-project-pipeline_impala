import datetime
import subprocess
from time import time
from matplotlib import pyplot as plt
import numpy as np
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tests.avan.test.create_yaml_tool import create_yaml

import optuna

Total_CPUs = 20
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

def get_throughput(files):
    prepare_times_per_time = []
    mean_explorer_times = []
    mean_train_times = []
    step_per_second = []
    for file in files:
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []
        mean_step_per_second_this_env_num = []
        mean_loop_times_this_env_num = []
        train_reward_avgs_this_env_num = []

        total_skip = 2
        skip_exp = 0
        skip_train = 0
        with open(file, "r") as f:
            lines=f.readlines()
            for line in lines:  
                # print(line)
                if line == "\n":
                    continue 
                key = line.split("\t")[0]
                val = line.split("\t")[1].strip()
                if key.startswith("mean_explore_ms") and val != "nan":
                    if skip_exp < total_skip:
                        skip_exp += 1
                        continue
                    mean_explorer_times_this_env_num.append(float(val))
                if key.startswith("mean_train_time_ms") and val != "nan":
                    if skip_train < total_skip:
                        skip_train += 1
                        continue
                    # if float(val) < 100:
                    mean_train_times_this_env_num.append(float(val))
                if key.startswith("step_per_second") and val != "nan":
                    if skip_train < total_skip:
                        skip_train += 1
                        continue
                    # if float(val) < 100:
                    mean_step_per_second_this_env_num.append(float(val))

                if key.startswith("mean_loop_time_ms") and val != "nan":
                    if skip_train < total_skip:
                        skip_train += 1
                        continue
                    # if float(val) < 100:
                    mean_loop_times_this_env_num.append(float(val))

                if key.startswith("train_reward_avg") and val != "nan":
                    if skip_train < total_skip:
                        skip_train += 1
                        continue
                    # if float(val) < 100:
                    train_reward_avgs_this_env_num.append(float(val))

        mean_explorer_times.append(np.mean(mean_explorer_times_this_env_num))
        mean_train_times.append(np.mean(mean_train_times_this_env_num))
        step_per_second = mean_step_per_second_this_env_num
        mean_loop_time = mean_loop_times_this_env_num
        train_reward_avgs = train_reward_avgs_this_env_num
        # print(mean_explorer_times)
        # print(mean_train_times)
        # print("mean_loop_time ===================== {}".format(mean_loop_time))
        # print("train_reward_avgs ===================={}".format(train_reward_avgs))
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []
        # real_throughput = 128 / np.mean(mean_loop_time) * 1000
        # return np.mean(mean_loop_time), np.max(train_reward_avgs)
        return step_per_second[-1], np.max(train_reward_avgs)

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features

    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def fn(trail):
    # abs_path = "/sdb1_data/avan/xingtian-master3/xingtian-master/multi-objective/train_yaml/train_impala"
    abs_path = "/home/data/xys/test_mo/xingtian-master3/xingtian-master/multi-objective/train_yaml/train_impala"
    # file_name_ppolite = "/home/xys/xingtian-ppo-v1/breakout_ppo2.yaml"
    # file_name_impala = "/sdb1_data/avan/xingtian-master3/xingtian-master/breakout_impala.yaml"
    # file_name_impala = "/home/data/xys/test_mo/xingtian-master3/xingtian-master/multi-objective/breakout_impala.yaml"
    # file_name_ppo = "/home/data/xys/test_mo/xingtian-master3/xingtian-master/breakout_ppo.yaml"
    file_name_ppo = "/home/xys/xingtian-ppo-v1/beamrider_ppo_best.yaml"
    is_impala = False
    pipeline = True
    now_time = time()
    str_time = str(datetime.datetime.now().strftime('%F_%T'))
    # env_pool_size = trail.suggest_int("env_pool_size", 2, Total_CPUs - 2)
    # env_pool_wait_nums = trail.suggest_int('env_pool_wait_nums', env_pool_size // 2 + 1, env_pool_size - env_pool_size // 3)
    # if env_pool_wait_nums > env_pool_size:
        # env_pool_wait_nums = env_pool_size
    update_dic = {}
    # update_dic.update({"size": env_pool_size})
    # update_dic.update({"wait_nums": env_pool_wait_nums})
    # update_dic.update({"env_num": Total_CPUs - env_pool_size})
    update_dic.update({"now_time": now_time})
    # train_per_checkpoint = trail.suggest_int("train_per_checkpoint", 1, 1)
    if is_impala:
        vector_env_size = trail.suggest_int("vector_env_size", 1, 6)
        prepare_times_per_train = trail.suggest_int("prepare_times_per_train", 1, 4)
        grad_norm_clip = trail.suggest_float("grad_norm_clip", 10.0, 80.0, step=10.0)
        update_dic.update({"prepare_times_per_train": prepare_times_per_train})
        update_dic.update({"vector_env_size": vector_env_size})
        update_dic.update({"grad_norm_clip": grad_norm_clip})
        sample_batch_step = trail.suggest_int("sample_batch_step", 64, 192, step=64)
        update_dic.update({"sample_batch_step": sample_batch_step})


    else:
        CRITIC_LOSS_COEF = trail.suggest_float("CRITIC_LOSS_COEF", 0.4, 1.0, step=0.1)
        ENTROPY_LOSS = trail.suggest_float("ENTROPY_LOSS", 0.001, 0.011, step=0.001)
        LOSS_CLIPPING = trail.suggest_float("LOSS_CLIPPING", 0.05, 0.55, step=0.05)
        NUM_SGD_ITER = trail.suggest_int("NUM_SGD_ITER", 3, 10, step=1)
        gpu_num = trail.suggest_int("gpu_num", 1, 4, step=1)
        update_dic.update({"CRITIC_LOSS_COEF": CRITIC_LOSS_COEF})
        update_dic.update({"ENTROPY_LOSS": ENTROPY_LOSS})
        update_dic.update({"LOSS_CLIPPING": LOSS_CLIPPING})
        update_dic.update({"NUM_SGD_ITER": NUM_SGD_ITER})
        update_dic.update({"gpu_num": gpu_num})

    max_steps = trail.suggest_int("max_steps", 128, 1280, step=32)

    # batch_trajs = max_steps * vector_env_size * prepare_times_per_train

    System_BATCH_SIZE = trail.suggest_int("System_BATCH_SIZE", 128, 2048, step=128)
    System_LR = trail.suggest_float("System_LR", 0.0001, 0.0012, step=0.0001)

    update_dic.update({"max_steps": max_steps})
    update_dic.update({"BATCH_SIZE": System_BATCH_SIZE})
    update_dic.update({"LR": System_LR})
    # update_dic.update({"train_per_checkpoint": train_per_checkpoint})
    if pipeline:
        env_pool_size = trail.suggest_int("env_pool_size", 2, Total_CPUs - 2)
        env_pool_wait_nums = trail.suggest_int('env_pool_wait_nums', env_pool_size // 2 + 1, env_pool_size - env_pool_size // 3)
        if env_pool_wait_nums > env_pool_size:
            env_pool_wait_nums = env_pool_size
        update_dic = {}
        update_dic.update({"size": env_pool_size})
        update_dic.update({"wait_nums": env_pool_wait_nums})
        update_dic.update({"env_num": Total_CPUs - env_pool_size})
        update_dic.update({"now_time": now_time})

    file_path = create_yaml(file_name_ppo, update_dic, abs_path, str_time)
    print("filepath ================ {}".format(file_path))
    p = subprocess.Popen(
        '/home/xys/xingtian-ppo-v1/single_train.sh ' + file_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
        # break
    
    
    filename = "/tmp/"+str(now_time)
    print("filename ================== {}".format(filename))
    mean_loop_time, max_reward = get_throughput([filename])
    if is_impala:
        real_throughput = max_steps * prepare_times_per_train * vector_env_size / mean_loop_time * 1000
    else:
        real_throughput = max_steps * 24 / mean_loop_time * 1000

    print("real_throughput ============= {}".format(real_throughput))
    print("max_reward ============= {}".format(max_reward))

    # print("step_per_second ======== {}".format(step_per_second[-1]))
    # print(step_per_second[-1])
    return real_throughput, max_reward

# Defines training and evaluation.
def train_model(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES

    flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
    return flops, accuracy

def objective(trial):
    train_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=False, transform=torchvision.transforms.ToTensor()
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    model = define_model(trial).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )

    for epoch in range(10):
        train_model(model, optimizer, train_loader)
    flops, accuracy = eval_model(model, val_loader)
    return flops, accuracy

seed = 128
num_variables = 2
n_startup_trials = 11 * num_variables - 1

def main():

    sampler = optuna.samplers.MOTPESampler(
        n_startup_trials=n_startup_trials, n_ehvi_candidates=24, seed=seed
    )
    study = optuna.create_study(directions=["maximize", "maximize"], storage='sqlite:///db.sqlite3', sampler=sampler)
    # study.optimize(objective, n_trials=n_startup_trials + 10, timeout=300)
    study.optimize(fn, n_trials=31)
    # study.optimize(fn, n_trials=1,)


    print("Number of finished trials: ", len(study.trials))
    optuna.visualization.plot_pareto_front(study, target_names=["throughput", "rewards"])

def test_fn():
    start_0 = time()
    study = optuna.create_study(direction='maximize')
    study.optimize(fn, n_trials=1)
    print(study.best_params)
    print(study.best_value)
    end_0 = time()
    print("total time ============= {}".format(end_0 - start_0))

if __name__ == "__main__":
    main()
