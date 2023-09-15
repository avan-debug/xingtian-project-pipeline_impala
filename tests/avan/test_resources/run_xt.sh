mysql -u root -e "drop database example"
mysql -u root -e "CREATE DATABASE IF NOT EXISTS example"
optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/example"
./kill_xt.sh
time=`date +"%Y-%m-%d-%H-%M-%S"`
export tune_system=1
export with_samper=1
export iter=4
export trials=4
export tune_cost=1

log_path=run-$tune_system-$with_samper-$iter-$trials-$time.log
echo log/$log_path
python test_xt_with_tune.py >&log/$log_path

