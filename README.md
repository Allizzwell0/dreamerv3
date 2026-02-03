
# 训练

```bash
python dreamerv3/main.py \
  --task auv_custom \
  --configs auv \
  --logdir /home/mayue/logdir/auv/{timestamp} 
```

此外经常出现卡掉服务器崩现象，注意勤查看显卡状态

# 评估

目前评估函数写好，但之前训练模型与当前环境设计不匹配（加了导向角作为状态输入），故需要重新训练

```bash
python eval_auv.py \
  --ckpt /home/mayue/logdir/auv/{timestamp} \
  --episodes 200 \
  --out_dir /home/mayue/logdir/auv/{timestamp}/eval_output
```
可能出现问题找不到checkpoint，需要debug

完成评估后可绘图，得到测例轨迹图

```bash
python plot_auv.py \
  --csv /home/mayue/logdir/auv/{timestamp}/eval_output/trajectories.csv \
  --episode 0 \
  --success_threshold 1.2
```

# 训练时查看效果

```bash
python -m scope.viewer --basedir ~/logdir/auv --port 8000
```
后续可能多加入几项指标，实现当前轨迹跟踪图示展示


# PBT部分
确定reward结构后调参太多，采样PBT进行调参：
```bash
python pbt_auv.py
```

# 服务器走代理
由于远程配置clash太麻烦，采用本地挂梯子，然后SSH反向代理实现
具体配置已经写进了ssh文件，本机端口7897，服务器代理走7897就行
命令为：
```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
```

如果要在当前terminal取消代理：
```bash
unset http_proxy;
unset https_proxy;
```

# Baseline
为了对比修改后的dreamer效果，考虑其对比方法：pure_dreamer, LOS+ MPC, SAC

## SAC部分：
环境部分参数使用和dreamer相同的参数，主要成功判断依据为track_ratio，同时对比其他若干项目

```bash
python train_sac_auv.py \
  --logdir /home/mayue/logdir/sac_auv/run1 \
  --steps 20000000 \
  --n_envs 8 \
  --norm_obs \
  --tb /home/mayue/logdir/sac_auv/tb \
  --env moving_goal=True \
  --env max_goal_speed=2.0 \
  --env max_goal_turn_rate=1.2

tensorboard --logdir /home/mayue/logdir/tb_sac_auv

python SAC_base/eval_sac_auv.py \
  --model /home/mayue/logdir/sac_auv/run1/sac_final.zip \
  --vecnorm /home/mayue/logdir/sac_auv/run1/vecnormalize.pkl \
  --out_dir /home/mayue/logdir/sac_auv/run1/eval_output \
  --episodes 200 \
  --success_threshold 0.5 \
  --track_success_ratio 0.8

```

## traditional 部分

学习mpc部分，考虑使用线性还是非线性