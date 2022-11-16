batch_exp=2
batch_ensemble_exp=7
bash batch_ensemble_original_emnist.sh $batch_exp $batch_ensemble_exp resnet50
bash batch_ensemble_baselines_emnist.sh $batch_exp $batch_ensemble_exp resnet50
bash batch_ensemble_last_layer_emnist.sh $batch_exp $batch_ensemble_exp resnet50
bash batch_ensemble_dr_emnist.sh $batch_exp $batch_ensemble_exp PCA resnet50
bash batch_ensemble_dr_emnist.sh $batch_exp $batch_ensemble_exp Kernel_PCA resnet50
bash batch_ensemble_label_distritbution_emnist.sh $batch_exp $batch_ensemble_exp resnet50
bash batch_ensemble_avg_emnist.sh $batch_exp $batch_ensemble_exp mean_avg cuda:1 resnet50
bash batch_ensemble_avg_emnist.sh $batch_exp $batch_ensemble_exp fed_avg cuda:1 resnet50
bash batch_ensemble_layer_emnist.sh $batch_exp $batch_ensemble_exp 1 resnet50
bash batch_ensemble_layer_emnist.sh $batch_exp $batch_ensemble_exp 2 resnet50
bash batch_ensemble_layer_emnist.sh $batch_exp $batch_ensemble_exp 3 resnet50
bash batch_ensemble_layer_emnist.sh $batch_exp $batch_ensemble_exp -1 resnet50

cd ..
bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_baselines.sh
bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_last_layer.sh
bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_label_distribution.sh
bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_fed_avg.sh
bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_mean_avg.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_PCA.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_Kernel_PCA.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_layer_1.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_layer_2.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_layer_3.sh
#bash exp_results/shells/batch_ensemble_$batch_ensemble_exp\_layer_-1.sh