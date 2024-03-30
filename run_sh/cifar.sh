cd ..

python train_cifar.py \
    --model-dir  checkpoint/test   \
    --data-dir  /data/home/wzliu/z_project/data/   \
    --save-freq  100  \
    --dataset  cifar10  \
    --model  resnet18
