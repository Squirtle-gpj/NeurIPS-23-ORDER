set -x

# Beta 参数数组
#betas=(0.0001 0.0005 0.001 0.005 0.1 0.5 1)
#betas=(1)
#n_inner_step_per_outer_step=-1


script_name="main.py"
experiment_name="debug_distributional"
code_name="debug_main"
env_name="hopper-medium-v2"
seed=1234



# 获取当前时间戳
timestamp=$(date +%Y%m%d%H%M%S)




# 训练命令
training_commands=$(cat <<EOF
python3 ../${script_name} \
        --experiment_name ${experiment_name} \
        --code_name ${code_name} \
        --env_name ${env_name} \
        --seed ${seed}
EOF
)
    # 打印训练命令
  echo "$training_commands"
  # 执行训练命令
  if [[ ${1} != "slurm" ]]; then
      $training_commands
  fi




