from copy import deepcopy
import datetime
from multiprocessing import Lock
import os
from time import time
from venv import create
import yaml
from setproctitle import setproctitle

setproctitle("pbt")

def create_yaml(file_name, update_dic, abs_path, str_time):
    import yaml
    with open(file_name) as f:
        doc = yaml.safe_load(f)
    
    # doc["env_para"]["env_info"]["size"] = update_dic["size"]
    # doc["env_para"]["env_info"]["wait_nums"] = update_dic["wait_nums"]

    # doc["env_num"] = update_dic["env_num"]
    if "prepare_times_per_train" in update_dic:
        doc["alg_para"]["alg_config"]["prepare_times_per_train"] = update_dic["prepare_times_per_train"]
    doc["alg_para"]["alg_config"]["BATCH_SIZE"] = update_dic["BATCH_SIZE"]
    # doc["alg_para"]["alg_config"]["train_per_checkpoint"] = update_dic["train_per_checkpoint"]
    if "vector_env_size" in update_dic:
        doc["env_para"]["env_info"]["vector_env_size"] = update_dic["vector_env_size"]
    doc["agent_para"]["agent_config"]["max_steps"] = update_dic["max_steps"]
    if "sample_batch_step" in update_dic:
        doc["model_para"]["actor"]["model_config"]["sample_batch_step"] = update_dic["sample_batch_step"]
    if "grad_norm_clip" in update_dic:
        doc["model_para"]["actor"]["model_config"]["grad_norm_clip"] = update_dic["grad_norm_clip"]

    if "CRITIC_LOSS_COEF" in update_dic:
        doc["model_para"]["actor"]["model_config"]["CRITIC_LOSS_COEF"] = update_dic["CRITIC_LOSS_COEF"]
    if "ENTROPY_LOSS" in update_dic:
        doc["model_para"]["actor"]["model_config"]["ENTROPY_LOSS"] = update_dic["ENTROPY_LOSS"]
    if "LOSS_CLIPPING" in update_dic:
        doc["model_para"]["actor"]["model_config"]["LOSS_CLIPPING"] = update_dic["LOSS_CLIPPING"]
    if "CRITIC_LOSS_COEF" in update_dic:
        doc["model_para"]["actor"]["model_config"]["NUM_SGD_ITER"] = update_dic["NUM_SGD_ITER"]    
    if "gpu_num" in update_dic:
        doc["model_para"]["actor"]["model_config"]["gpu_num"] = update_dic["gpu_num"]
    
    if "size" in update_dic:
        doc["env_para"]["env_info"]["size"] = update_dic["size"]
        doc["env_para"]["env_info"]["wait_nums"] = update_dic["wait_nums"]
        doc["env_num"] = update_dic["env_num"]

    doc["model_para"]["actor"]["model_config"]["LR"] = update_dic["LR"]

    str_ = doc["model_para"]["actor"]["model_name"]
    str_filename = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
        str_filename += "_" + str(key) + "_" + str(int(val))
    
    doc["benchmark"].update({"id": str_ + "_" + str_time + "+"})
    doc["benchmark"].update({"archive_root": doc["benchmark"]["archive_root"]})

    if "now_time" in update_dic:
        doc["env_para"].update({"now_time": update_dic["now_time"]})

    with open(abs_path + str_filename + ".yaml", 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    return abs_path + str_filename + ".yaml"


lock = Lock()

def create_yaml_LR(file_name, update_dic, abs_path):
    lock.acquire()
    with open(file_name, "r+", encoding='utf-8') as f:
        doc = yaml.safe_load(f)

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        if key == "LR" or key == "now_time":
            str_ += "_" + str(key) + "_" + str(val)

    
    if "LR" in update_dic.keys():
        update_dic.update({"LR": float(update_dic["LR"])})

    doc["benchmark"].update({"id": str_})
    doc["agent_para"]["agent_config"].update(update_dic)
    doc["model_para"]["actor"]["model_config"].update(update_dic)
    doc["env_para"]["env_info"].update(update_dic)
    doc["env_para"].update({"now_time": update_dic["now_time"]}) 
    
    doc.update(update_dic)      
    yaml_filepath = abs_path + doc["benchmark"]["id"] + ".yaml"
    bench_filepath = os.path.join(doc["benchmark"]["archive_root"], str_)

    try:
        
        with open(yaml_filepath, 'w+', encoding='utf-8') as f:
            yaml.dump(doc, f)
        lock.release()
    except:
        print("exception ===================== doc =============== {}".format(doc))        
        with open(yaml_filepath, 'w+', encoding='utf-8') as f:
            doc2 = {}
            doc2.update(doc)
            yaml.dump(doc2, f, default_flow_style=False)
        lock.release()
        # print("exception ===================== doc =============== {}".format(doc))
        # print("exception ===================== type(doc) =============== {}".format(type(doc)))
        # doc2 = deepcopy(doc)
        # with open(yaml_filepath, 'w+') as f:
        #     yaml.safe_dump(doc2, f)
        
    return yaml_filepath, bench_filepath

# if __name__ == "__main__":
#     abs_path = "/home/xys/xingtian-test/xingtian-master3/xingtian-master/train_yaml_test"
#     os.mkdir(abs_path)
#     str_time = str(datetime.datetime.now().strftime('%F_%T'))
#     # file_name_ppo = "/home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo.yaml"
#     file_name_ppolite = "/home/xys/xingtian-ppo-v1/train_yaml/breakout_ppo2.yaml"

#     update_arrs = []
#     update_dic1 = {"size": 4, "wait_nums": 1, "env_num": 4}
#     update_dic2 = {"size": 5, "wait_nums": 1, "env_num": 3}
#     update_dic3 = {"size": 6, "wait_nums": 2, "env_num": 2}
#     update_dic4 = {"size": 5, "wait_nums": 2, "env_num": 5}
#     update_dic5 = {"size": 5, "wait_nums": 1, "env_num": 5}
#     update_dic6 = {"size": 6, "wait_nums": 2, "env_num": 4}
#     update_dic7 = {"size": 6, "wait_nums": 1, "env_num": 4}

    

#     update_arrs.append(update_dic1)
#     update_arrs.append(update_dic2)
#     update_arrs.append(update_dic3)
#     update_arrs.append(update_dic4)
#     update_arrs.append(update_dic5)
#     update_arrs.append(update_dic6)
#     update_arrs.append(update_dic7)
#     # update_arrs.append(update_dic8)

#     for dic in update_arrs:
#         create_yaml(file_name_ppolite, dic, abs_path, str_time)
    # for dic in update_arrs:
    #     create_yaml(file_name_ppolite, dic, abs_path, str_time)
    # update_dic6 = {"LR": 0.00015}
    # update_dic5 = {"LR": 0.00020}
    # update_dic1 = {"LR": 0.00025}
    # update_dic2 = {"LR": 0.00030}
    # update_dic3 = {"LR": 0.00010}
    # update_dic2 = {"LR": 0.00055}
    # update_dic4 = {"LR": 0.00005}
    # update_dic3 = {"LR": 0.00075}
    # update_dic1 = {"LR": 0.0001}
    # update_dic2 = {"LR": 0.00025}
    # update_dic3 = {"LR": 0.0005}
    # update_dic4 = {"LR": 0.00075}
    # update_dic8 = {"LR": 0.001}
    # update_dic5 = {"LR": 0.00125}
    # update_dic6 = {"LR": 0.0015}
    # update_dic7 = {"LR": 0.00175}

    # update_arrs.append(update_dic1)
    # update_arrs.append(update_dic2)
    # update_arrs.append(update_dic3)
    # update_arrs.append(update_dic4)
    # update_arrs.append(update_dic5)
    # update_arrs.append(update_dic6)
    # update_arrs.append(update_dic3)
    # update_arrs.append(update_dic4)
    # update_arrs.append(update_dic5)
    # update_arrs.append(update_dic6)
    # update_arrs.append(update_dic7)
    # update_arrs.append(update_dic8)

    # for dic in update_arrs:
    #     create_yaml_LR(file_name_ppolite, dic, abs_path)


# 8: 4-2-4
# 10: 6-3-4
# 6: 






