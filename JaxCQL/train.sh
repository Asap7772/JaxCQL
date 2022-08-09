export WANDB_API_KEY='YOUR W&B API KEY HERE'

gpus=(0 1 2 3 4 5 6 7)
envs=(halfcheetah-medium-v2 halfcheetah-medium-replay-v2 \
    halfcheetah-medium-expert-v2 hopper-medium-v2 \
    hopper-medium-replay-v2 hopper-medium-expert-v2 \
    walker2d-medium-v2 walker2d-medium-replay-v2 \
    walker2d-medium-expert-v2)

output_dir='experiments_divergence'
current_dir=$(pwd)
full_output_dir=$current_dir/$output_dir
dry_run=true

mkdir -p $full_output_dir

index=0
for env in ${envs[@]}; do
    which_gpu=${gpus[$index]}
    export CUDA_VISIBLE_DEVICES=${gpus[which_gpu]}

    echo "Training ${env} on GPU ${gpus[which_gpu]}"

    command="python -m JaxCQL.conservative_sac_main \
    --env $env \
    --logging.output_dir $full_output_dir \
    --logging.online"

    if $dry_run; then
        echo $command
    else
        $command
    fi

    index=$((index+1))
done
