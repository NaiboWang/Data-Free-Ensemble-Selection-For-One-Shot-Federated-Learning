batch_exp=27
batch_ensemble_exp=31
bash batch_ensemble_original_cifar100.sh $batch_exp $batch_ensemble_exp dla
bash batch_ensemble_baselines_cifar100.sh $batch_exp $batch_ensemble_exp dla
bash batch_ensemble_last_layer_cifar100.sh $batch_exp $batch_ensemble_exp dla
bash batch_ensemble_dr_cifar100.sh $batch_exp $batch_ensemble_exp PCA dla
bash batch_ensemble_dr_cifar100.sh $batch_exp $batch_ensemble_exp Kernel_PCA dla
bash batch_ensemble_label_distritbution_cifar100.sh $batch_exp $batch_ensemble_exp dla
bash batch_ensemble_avg_cifar100.sh $batch_exp $batch_ensemble_exp mean_avg cuda:1 dla
bash batch_ensemble_avg_cifar100.sh $batch_exp $batch_ensemble_exp fed_avg cuda:1 dla
bash batch_ensemble_layer_cifar100.sh $batch_exp $batch_ensemble_exp 1 dla
bash batch_ensemble_layer_cifar100.sh $batch_exp $batch_ensemble_exp 2 dla
bash batch_ensemble_layer_cifar100.sh $batch_exp $batch_ensemble_exp 3 dla
bash batch_ensemble_layer_cifar100.sh $batch_exp $batch_ensemble_exp -1 dla

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