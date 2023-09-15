import os
import subprocess
import time

import yaml
from test_mo import get_throughput
from ray.air import session, Checkpoint

def create_yaml_LR(file_name, update_dic, abs_path):
    with open(file_name) as f:
        doc = yaml.safe_load(f)

    str_ = doc["model_para"]["actor"]["model_name"]
    for key, val in update_dic.items():
        str_ += "_" + str(key) + "_" + str(val)
    doc["benchmark"].update({"id": str_})

    doc["model_para"]["actor"]["model_config"].update(update_dic)
    doc["env_para"].update({"now_time": update_dic["now_time"]})

    bench_filepath = os.path.join(doc["benchmark"]["archive_root"], str_)

    yaml_filepath = abs_path + doc["benchmark"]["id"] + ".yaml"
    with open(yaml_filepath, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)
    
    return yaml_filepath, bench_filepath

def train_convnet(config):
    # Create our data loaders, model, and optmizer.
    step = 1
    # train_loader, test_loader = get_data_loaders()
    # model = ConvNet()
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config.get("lr", 0.01),
    #     momentum=config.get("momentum", 0.9),
    # )

    yaml_file_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yamls/breakout_impala_origin.yaml"
    str_time = str(time.time()).replace(".", "")
    update_dic = {}

    update_dic.update({"LR": round(config.get("lr", 0.01), 6)})
    update_dic.update({"now_time": str_time})
    abs_path = "/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_test"
    # update_dic.update({"LR": config.get("lr", 0.01)})
    yaml_file_path, bench_filepath = create_yaml_LR(yaml_file_path, update_dic, abs_path)

    print("yaml_file_path ================= {}".format(yaml_file_path))
    print("bench_filepath ================= {}".format(bench_filepath))

    model_filepath = os.path(bench_filepath, "models/actor")
    print("model_filepath ================= {}".format(model_filepath))
    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.
    if session.get_checkpoint():
        # Load model state and iteration step from checkpoint.
        checkpoint_dict = session.get_checkpoint().to_dict()

        update_dic.update({"init_weights": checkpoint_dict["model_filepath"]})

        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

    while True:
        p = subprocess.Popen(
            '/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/test/single_train.sh ' + yaml_file_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout.readlines():
            print(line)
        
        filename = "/tmp/"+str(str_time)
        throughput, reward = get_throughput([filename])

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
    
train_convnet({"LR" : 1.0})