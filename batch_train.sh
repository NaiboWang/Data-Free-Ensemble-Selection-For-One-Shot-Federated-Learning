# 训练模型
config_emnist_digits=(
201 # batch_size
noniid-\#label3
"emnist" # dataset
'digits' # split
10 # num_classes
1 # input_channels
#[5,10,100,200,400]
)
config_emnist_letters=(
201 # batch_size
noniid-\#label8
"emnist" # dataset
'letters' # split
26 # num_classes
1 # input_channels
#[5,10,100,200,400]
)
config_emnist_balanced=(
201 # batch_size
noniid-\#label18
"emnist" # dataset
'balanced' # split
47 # num_classes
1 # input_channels
#[5,10,100,200,400]
)
config_cifar10=(
211 # batch_size
noniid-\#label4
"cifar10" # dataset
'cifar10' # split
10 # num_classes
3 # input_channels
#[5,10,50,100,200]
)
config_cifar100=(
221 # batch_size
noniid-\#label45
"cifar100" # dataset
'cifar100' # split
100 # num_classes
3 # input_channels
#[5,10,20]
)
# Modify this part every time
party_num=200
device="cuda:1"
config=("${config_emnist_digits[@]}")
model="SpinalNet"

batch=${config[0]}
partition=(
${config[1]}
homo
iid-diff-quantity
noniid-labeldir
)
dataset=${config[2]}
split=${config[3]}
num_classes=${config[4]}
input_channels=${config[5]}


#echo $(seq 0 $[party_num - 1])
for element in ${partition[@]} # 遍历数组
#也可以写成for element in ${array[*]}
do
  for i in $(seq 0 $[party_num - 1])
  do
  	python3 train_model.py --index $i --partition $element --party_num $party_num --split $split --device $device --batch $batch --dataset $dataset --model $model --input_channels $input_channels --num_classes $num_classes
  echo "Finish of party "$i
  done
  python3 train_model.py --index -1 --partition $element --party_num $party_num --split $split --device $device --batch $batch --dataset $dataset --model $model --input_channels $input_channels --num_classes $num_classes
done
