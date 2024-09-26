#! /bin/bash

# simple-search.sh

LOGDIR=logs/run1
#SEED=123
DB=bidask-f1-c100
NENVS=16
env_done_fn=2
env_start_fn=2
POLICY=cnn
ENV=lim-discrete-cancel

#TODO - compare rew3 with and without time scaling
for env_state_lvls in 51; do
for env_state_w in 2 5; do
for env_freq in 5; do
for env_reward_fn in 3; do #1 2 3 4; do
for env_cum_state in True False; do
#for env_normalize in True False; do

for model_nsteps in 128 512; do
for model_nminibatches in 4 16; do
for model_gamma in 0.999; do # 0.99 0.999
for model_noptepochs in 4 8; do
for model_ent_coef in 0.0 0.001; do
for model_vf_coef in 0.1 0.5; do
for model_lr in 2e-4 3e-4; do
for model_cliprange in 0.1 0.2 0.3; do
    for SEED in {1..5}; do
        ARGS="--env=$ENV --policy=$POLICY --db=$DB --nenvs=$NENVS --env.state_lvls=$env_state_lvls --env.state_w=$env_state_w --env.freq=$env_freq --env.reward_fn=$env_reward_fn --env.cum_state=$env_cum_state  --env.normalize=$env_normalize --env.done_fn=$env_done_fn --env.start_fn=$env_start_fn
        --model.nsteps=$model_nsteps --model.nminibatches=$model_nminibatches --model.gamma=$model_gamma --model.noptepochs=$model_noptepochs --model.ent_coef=$model_ent_coef --model.vf_coef=$model_vf_coef --model.lr=$model_lr --model.cliprange=$model_cliprange"
        echo $ARGS --seed $SEED
        PYTHONPATH=. python3 deeptrade/run_openai.py $ARGS --seed $SEED --logdir $LOGDIR &
        #aws s3 sync logs/ s3://deeptrade.logs/gridsearch/
    done
    #sleep 1
    wait
done
done
done
done
done
done
done
done
done
done
done
done
done
done
