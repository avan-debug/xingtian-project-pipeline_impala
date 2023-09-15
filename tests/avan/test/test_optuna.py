import optuna
import yaml
 
def objective(trial):
    x = trial.suggest_uniform('x', 0, 1)
    y = trial.suggest_uniform('y', 0, 1)
    z = trial.suggest_int("z",0, 6)
    if x + y > 0.5:
        return -1
    return (x + y + z) ** 2
 
study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)

study1 = optuna.create_study(direction='maximize')

study1.optimize(objective, n_trials=2)
print(study1.best_params)
print(study1.best_value)

# print("===================================")
# study.optimize(objective, n_trials=10)
# print(study.best_params)
# print(study.best_value)
class A:
     def __init__(self, i) -> None:
          self.i = i

# a = A(0.01)
# # a = []
# doc = {'alg_para': {'alg_name': 'IMPALAOpt', 'alg_config': {'train_per_checkpoint': 1, 'prepare_times_per_train': 1, 'BATCH_SIZE': 512, 'save_model': True, 'save_interval': 5}}, 'env_para': {'now_time': '16939879098166382', 'env_name': 'VectorAtariEnv', 'env_info': {'name': 'BreakoutNoFrameskip-v4', 'vision': False, 'vector_env_size': 1, 'LR': 0.0008915061871200673, 'now_time': '16939879098166382', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0008915061871200673_now_time_1693987739031112/models/actor'}}, 'agent_para': {'agent_name': 'AtariImpalaOpt', 'agent_num': 1, 'agent_config': {'max_steps': 128, 'complete_step': 200000}}, 'model_para': {'actor': {'model_name': 'ImpalaCnnOpt', 'state_dim': [84, 84, 4], 'input_dtype': 'uint8', 'state_mean': 0.0, 'state_std': 255.0, 'action_dim': 4, 'model_config': {'LR': 0.0008915061871200673, 'sample_batch_step': 128, 'grad_norm_clip': 40.0, 'now_time': '16939879098166382', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0008915061871200673_now_time_1693987739031112/models/actor'}}}, 'env_num': 10, 'benchmark': {'id': 'ImpalaCnnOpt_LR_0.0008915061871200673_now_time_16939879098166382', 'archive_root': '/home/xys/avan/xt_archive', 'log_interval_to_train': 2}, 'LR': 0.0008915061871200673, 'now_time': '16939879098166382', 'init_weights': '/home/xys/avan/xt_archive/ImpalaCnnOpt_LR_0.0008915061871200673_now_time_1693987739031112/models/actor'}
# print(doc["alg_para"])
# print(type(doc))
# # try:
# with open("/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_testImpalaCnnOpt_LR_1e-04_now_time_16939854699092922.yaml", 'w+') as f:
#         yaml.dump(a, f)
# except:
#      print("god        ")
#      with open("/home/xys/xingtian-test/xingtian-project-pipeline_impala/tests/avan/train_yaml/train_yaml_imopt/train_yaml_testImpalaCnnOpt_LR_1e-04_now_time_16939854699092922.yaml", 'w+') as f:
#             yaml.safe_dump(doc, f, default_flow_style=False)


