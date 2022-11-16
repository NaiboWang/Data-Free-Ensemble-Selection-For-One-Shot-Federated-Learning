# 训练模型
batch=3
partition=(
noniid-\#label8
homo
iid-diff-quantity
noniid-labeldir
)
party_num=10
device="cuda:0"
dataset="emnist"
split="letters"
num_classes=26
model="resnet50"
input_channels=1

#echo $(seq 0 $[party_num - 1])
for element in ${partition[@]} # 遍历数组
#也可以写成for element in ${array[*]}
do
  for i in $(seq 0 $[party_num - 1])
  do
  	python3 EMNIST_VGG_and_SpinalVGG.py --index $i --partition $element --party_num $party_num --split $split --device $device --batch $batch --dataset $dataset --model $model --input_channels $input_channels --num_classes $num_classes
  echo "Finish of party "$i
  done
  python3 EMNIST_VGG_and_SpinalVGG_Oracle.py --index -1 --partition $element --party_num $party_num --split $split --device $device --batch $batch --dataset $dataset --model $model --input_channels $input_channels --num_classes $num_classes
done



