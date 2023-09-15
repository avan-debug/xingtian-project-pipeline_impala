kill -9 $(pgrep xt_explorer)
kill -9 $(pgrep xt_main)
kill -9 $(pgrep xt_broker)
kill -9 $(pgrep xt_predictor)
kill -9 $(pgrep plasma-store-se)
kill -9 $(pgrep xt_compress)
kill -9 $(pgrep multi_trainer)
# pkill -9 python3
pkill -9 xt
# pkill -9 pbt

