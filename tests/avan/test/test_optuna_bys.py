import datetime
import time
import subprocess
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from tests.avan.test.create_yaml_tool import create_yaml, create_yaml_LR
import optuna
import math
from multiprocessing import Lock

raw_file_path = " "
run_log = "/tmp/test_opt_bys.log"
Total_CPUs = 10


def get_throughput(files):
    prepare_times_per_time = []
    mean_explorer_times = []
    mean_train_times = []
    step_per_second = []
    for file in files:
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []
        mean_step_per_second_this_env_num = []
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

        mean_explorer_times.append(np.mean(mean_explorer_times_this_env_num))
        mean_train_times.append(np.mean(mean_train_times_this_env_num))
        step_per_second = mean_step_per_second_this_env_num
        print(mean_explorer_times)
        print(mean_train_times)
        mean_explorer_times_this_env_num = []
        mean_train_times_this_env_num = []
        return step_per_second[-1]

# raw_file_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml"

def fn(trail):
    abs_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_test"
    now_time = str(time.time()).replace(".", "")
    env_pool_size = trail.suggest_int("env_pool_size", 3, Total_CPUs - 2)
    env_pool_wait_nums = trail.suggest_int('env_pool_wait_nums', env_pool_size // 2 + 1, env_pool_size - env_pool_size // 3)
    # group_num = trail.suggest_int('group_num', 1, 4)
    # infer_size = trail.suggest_int('infer_size', 2, env_pool_size)
    infer_size = Total_CPUs - env_pool_size
    # prepare_times_per_train = trail.suggest_int('prepare_times_per_train', 1, 4)
    # max_steps = trail.suggest_int('max_steps', 1, 4)
    # sample_batch_step = trail.suggest_int('sample_batch_step', 1, 4)
    # sample_batch_step = sample_batch_step * 128
    # max_steps = max_steps * 128
    group_num = 1
    env_num = infer_size * group_num
    if env_pool_wait_nums > env_pool_size:
        env_pool_wait_nums = env_pool_size
    update_dic = {}
    update_dic.update({"size": env_pool_size})
    update_dic.update({"vector_env_size": env_pool_size})
    update_dic.update({"wait_num": env_pool_wait_nums})
    update_dic.update({"env_num": env_num})
    # update_dic.update({"group_num": group_num})
    # update_dic.update({"prepare_times_per_train": prepare_times_per_train})
    # update_dic.update({"max_steps": max_steps})
    # update_dic.update({"sample_batch_step": sample_batch_step})
    update_dic.update({"now_time": now_time})
    
    complete_step = 15000
    update_dic.update({"complete_step": complete_step})

    yaml_filepath, bench_filepath = create_yaml_LR(raw_file_path, update_dic, abs_path)
    print("yaml_filepath ================== {}".format(yaml_filepath))
    run_log = "/tmp/test_opt_bys" + now_time + ".log"
    print("run_log ======================== {}".format(run_log))
    p = subprocess.Popen(
        '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + yaml_filepath + " " + run_log, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
        break
    
    print("log filename ================== {}".format(filename))

    filename = "/tmp/"+str(now_time)
    step_per_second = get_throughput([filename])
    print(step_per_second)
    # if step_per_second < 500:
    #     return 10000
    # return (env_pool_size + infer_size) * group_num / step_per_second
    return step_per_second


def find_max_th(yaml_filepath, study, n_trials):
    global raw_file_path
    raw_file_path = yaml_filepath
    start_0 = time.time()
    
    study.optimize(fn, n_trials=n_trials)
    print(study.best_params)
    print(study.best_value)
    end_0 = time.time()
    print("get best throughput total time ============= {}".format(end_0 - start_0))
    return study.best_params

# if __name__ == "__main__":
#     study = optuna.create_study(direction='maximize')
#     find_max_th("/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_pipeline_128opt.yaml", study, 3)

    # 8-6


