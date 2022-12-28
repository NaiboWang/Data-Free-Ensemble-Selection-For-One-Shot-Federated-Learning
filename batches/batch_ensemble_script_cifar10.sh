batch=$1
batch_ensemble=$2
# dr="noDimensionReduction"
model=$3
device=$4
# label_distribution=$5
selection_method="mixed"
# last_layer=$7
# layer=$8
# name=$9
normalization=1
avg="none"
input_channels=3
num_classes=10

# baselines
bash batch_ensemble_baselines_cifar10.sh $batch $batch_ensemble $model
# last_layer
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 1 0 "last_layer"
# fed_avg and mean_avg
bash batch_ensemble_avg_cifar10.sh $batch $batch_ensemble fed_avg $device $model
bash batch_ensemble_avg_cifar10.sh $batch $batch_ensemble mean_avg $device $model

# label_distribution last_layer
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 1 $selection_method 1 0 "last_layer_label_distribution"
# PCA
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble PCA $model 0 $selection_method 1 0 "PCA"
# Kernel_PCA
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble Kernel_PCA $model 0 $selection_method 1 0 "Kernel_PCA"
# layer 1
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 0 1 "layer_1"
# layer 2
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 0 2 "layer_2"
# layer 3
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 0 3 "layer_3"
# layer -1
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 0 -1 "layer_-1"
# CV Selection Method
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 "CV" 1 0 "last_layer_CV"
# data Selection Method
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 "data" 1 0 "last_layer_data"
# original
bash batch_ensemble_clustering_cifar10.sh $batch $batch_ensemble noDimensionReduction $model 0 $selection_method 0 0 "original"

if [ $5 -gt 0 ] 
then
  cd ..
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_baselines.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_last_layer.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_last_layer_label_distribution.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_PCA.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_Kernel_PCA.sh
  if [ $5 -eq 2 ] # If the 5th command line if equal to 2 then run fed and mean avg
  then
    bash exp_results/shells/batch_ensemble_$batch_ensemble\_fed_avg.sh
    bash exp_results/shells/batch_ensemble_$batch_ensemble\_mean_avg.sh
  fi
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_PCA.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_Kernel_PCA.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_layer_1.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_layer_2.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_layer_3.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_layer_-1.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_last_layer_CV.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_last_layer_data.sh
  bash exp_results/shells/batch_ensemble_$batch_ensemble\_original.sh
fi