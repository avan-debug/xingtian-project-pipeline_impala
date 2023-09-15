import os
import subprocess

from multiprocessing import Process
import time

from subprocess import PIPE, Popen
from setproctitle import setproctitle
import optuna

def constraints(trial):
    return trial.user_attrs["constraint"]


sampler = optuna.integration.BoTorchSampler(
    constraints_func=constraints,
    n_startup_trials=1,
)


def run_xt():
    # p = subprocess.Popen(
    #             '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # for line in p.stdout.readlines():
    #     print(line)
    #     break

    # p = subprocess.Popen(
    #             'xt_main -f ' + "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.system('xt_main -f ' + "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml")
    # for line in p.stdout.readlines():
    #     print(line)
        # break
    print("good")
# run_xt()
ps = []
ps_num = 4

# p = Process(target=run_xt)
# p.start()
# p.join()

# for i in range(ps_num):
#     p = Process(target=run_xt)
#     ps.append(p)

# for i in range(ps_num):
#     ps[i].start()
#     time.sleep(20)

# for i in range(ps_num):
#     ps[i].join()
import yaml
import numpy as np
def test_write_yaml():
    a = np.array(1)

    lr = {"LR": float(9.999999747378752e-05)}
    p = lr.get("good", 0.01234234215463246234615235235)
    print(type(p))
    doc = {'agent_para': {'agent_config': {'complete_step': 80000, 'max_steps': 128}, 'agent_name': 'AtariImpalaOpt', 'agent_num': 1}, 'alg_para': {'alg_config': {'BATCH_SIZE': 512, 'prepare_times_per_train': 1, 'train_per_checkpoint': 1, 'save_model': True, 'save_interval': 50}, 'alg_name': 'IMPALAOpt'}, 'benchmark': {'id': 'ImpalaCnnOpt_LR_0.00086271705_now_time_16940634968704908', 'archive_root': '/home/xys/avan/xt_archive', 'log_interval_to_train': 2}, 'env_num': 10, 'env_para': {'env_info': {'name': 'BreakoutNoFrameskip-v4', 'size': 5, 'vector_env_size': 5, 'vision': False, 'wait_num': 3, 'LR': 0.00086271705, 'now_time': '16940634968704908', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0003266941025144861_now_time_16940632874591143/models/actor'}, 'env_name': 'EnvPool', 'now_time': '16940634968704908'}, 'group_num': 1, 'model_para': {'init_weight': '/home/xys/avan/xt_archive/xt_breakout+230905115413T0/models/actor01000.data-00000-of-00001', 'actor': {'action_dim': 4, 'input_dtype': 'uint8', 'model_config': {'LR': 0.00086271705, 'grad_norm_clip': 40.0, 'sample_batch_step': 128, 'now_time': '16940634968704908', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0003266941025144861_now_time_16940632874591143/models/actor'}, 'model_name': 'ImpalaCnnOpt', 'state_dim': [84, 84, 4], 'state_mean': 0.0, 'state_std': 255.0}}, 'speedup': 3, 'LR': 0.00086271705, 'now_time': '16940634968704908', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0003266941025144861_now_time_16940632874591143/models/actor'}

    with open("/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_testImpalaCnnOpt_LR_0.00086271705_now_time_16940634968704908.yaml1", "w+", encoding="utf-8") as f:
        yaml.dump(lr, f)

    doc = {}
    doc.keys()

def test_kill_plasma():
    setproctitle("fa")
    p = Process(target=run_plasma)
    p.start()
    p.join()

def run_plasma():
    setproctitle("run_plasma")
    os.setpgid(os.getpid(), os.getpid())
    print("os.getpgid(0) =============== {}".format(os.getpgid(0)))
    # p = Popen("plasma_store -m 1000000000 -s /tmp/plasma", shell=True, stderr=subprocess.STDOUT)
    os.system("plasma_store -m 1000000000 -s /tmp/plasma")
    print("goodd")

    # for line in p.stdout.readlines():
    #     print(line)

# test_kill_plasma()
# time.sleep(10)
# def real_kill():
#     os.system("pkill -9 run_plasma")
# print("begin kill")
# real_kill()
# print("over")

# os.system("plasma_store -m 1000000000 -s /tmp/plasma")

# 定义objective要优化的函数，最小化目标函数(x - 2)^2
def objective(trial):
    # 定义超参数的搜索空间，x的空间是-10到10之间的浮点数
    x = trial.suggest_float('x', -10, 10)
    print("trial.number ======================= {}".format(trial.number))
    return -(x - 2) ** 2

def run_optuna():
    sampler._candidates_fun
    study = optuna.load_study(
        study_name="distributed-example", 
        storage="mysql://root:tanklab@localhost/example",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=10)
    print("10 ============= {}".format(study.best_params))
    # study.optimize(objective, n_trials=0)
    # print("0 ============= {}".format(study.best_params))
    # 输出搜索出的最佳参数
# 创建一个study对象并调用该optimize方法超过 100 次试验
# study = optuna.create_study()
ps = []
ps_num = 4
for i in range(ps_num):  
    p = Process(target=run_optuna)
    ps.append(p)

for i in range(ps_num):
    ps[i].start()

    
for i in range(ps_num):
    ps[i].join()

# a = {"a": "b"}
# print(a)
# if "a" in a.keys():
#     print("good")
# del a["a"]
# print(a)
# 输出搜索出的最佳参数
# print(study.best_params)




