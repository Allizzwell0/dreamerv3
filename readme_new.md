
# 训练

```bash
python dreamerv3/main.py \
  --task auv_custom \
  --configs auv \
  --logdir /home/mayue/logdir/auv/{timestamp} \
```

此外经常出现卡掉服务器崩现象，注意勤查看显卡状态

# 评估

目前评估函数写好，但之前训练模型与当前环境设计不匹配（加了导向角作为状态输入），故需要重新训练

```bash
c
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