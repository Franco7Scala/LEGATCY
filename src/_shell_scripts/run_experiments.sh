#!/bin/bash
gpus=(4 5 6 7)

debug="True"
dataset_name="imdb"
n_snapshot=4
times_fist_snapshot=3
metapaths_enabled="False"
subgraph_hops=2
training_strategyies=(DyHANE ActiveERS2 FullRetraining OnlineTraining VotingStrategy)
sampling_techniques=(ActiveERS2 RandomALTechnique LCSALTechnique EntropyALTechnique MarginALTechnique None)
k=500                                    # used only with ActiveERS2 and VotingStrategy, it identifies the amount of data to keep from the old nodes
reduction_factor=0.5


source /home/scala/.virtualenvs/GNNContinualLearning/bin/activate
export PYTHONPATH=$PYTHONPATH:/projects/GNNContinualLearning/


for i in "${!ks[@]}"; do
  k="${ks[$i]}"
  path_active_technique="./results/${model_name//\//_}_${dataset_name//\//_}/active_technique/${selection_policy}/k_${k//./_}"
  mkdir -p $path_active_technique
  echo "Running Active Experiment with model: $model_name, dataset: $dataset_name, k value: $k, selection_policy: $selection_policy on GPU: $gpu_id_index"
  CUDA_VISIBLE_DEVICES=$gpu_id_index python src/main_train_active_technique.py --model_name "$model_name" --dataset_name "$dataset_name" --selection_policy "$selection_policy" --percentage_to_select "$k" --n_cycles "$n_cycles" --output_dir $path_active_technique --benchmarks $benchmarks --alignment_policy "$alignment_policy" --seed "$seed" > "$path_active_technique/log_$(date +%s%3N).log" 2>&1 &
  sleep 1
  gpu_id_index=${gpus[$(( (i + 1) % ${#gpus[@]} ))]}
done

wait

echo "Completed!"
