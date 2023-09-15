import datetime
from time import time
import subprocess
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
# import example as ex
from tests.avan.test.create_yaml_tool import create_yaml_LR
import yaml

Total_CPUs = 20

fspace = {
    "env_pool_size": hp.choice('env_pool_size', range(1, Total_CPUs)),
    "env_pool_wait_nums": hp.choice('env_pool_wait_nums', range(1, Total_CPUs))
}

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
        return step_per_second

def fn():
    abs_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_test"
    raw_file_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml"
    now_time = time()
    str_time = str(datetime.datetime.now().strftime('%F_%T'))
    # env_pool_size = int(params['env_pool_size'])
    # env_pool_wait_nums = int(params['env_pool_wait_nums'])
    # if env_pool_wait_nums > env_pool_size:
    #     env_pool_wait_nums = env_pool_size
    # if env_pool_wait_nums <= env_pool_size // 2:
    #     env_pool_wait_nums = (env_pool_size // 2) + 1
    update_dic = {}
    # update_dic.update({"size": env_pool_size})
    # update_dic.update({"wait_nums": env_pool_wait_nums})
    # update_dic.update({"env_num": Total_CPUs - env_pool_size})+
    update_dic.update({"now_time": now_time})
    yaml_file_path, bench_filepath = create_yaml_LR(raw_file_path, update_dic, abs_path)

    p = subprocess.Popen(
        '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + yaml_file_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout.readlines():
        print(line)
        break
    
    
    filename = "/tmp/"+str(now_time)
    step_per_second = get_throughput([filename])
    print(step_per_second[-1])
    return {"loss": -step_per_second[-1],'status': STATUS_OK}

if __name__ == "__main__":
    trials = Trials()
    start = time()
    best = fmin(
        fn=fn,
        space=fspace,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials)
    # params = {"env_pool_size": 10, "env_pool_wait_nums":6}
    print("best:")
    print(best)
    print('trials:')
    for trial in trials.trials:
        print(trial)
    end = time()
    print("total time ================= {}".format(end - start))

