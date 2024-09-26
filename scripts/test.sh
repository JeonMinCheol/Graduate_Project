# for aggregator in fedavg fedprox fednova
# do

# for selection in wait score
# do

for dataset in cifar10 FashionMNIST MNIST
do


python -u ../run.py \
 --model_name cnn \
 --data_root ../datasets/ \
 --data $dataset \
 --aggregator fedavg \
 --n_clients 2 \
 --frac 0.1 \
 --rounds 10 \
 --selection wait \
 --n_client_epochs 10 \
 --optim sgd \
 --lr 0.005 \
 --momentum 0.9 \
 --log_every 1 \
 --early_stopping 1 \
 --use_multiple_gpu \
 --use_gpu \
 --gpu 2

done

#done

#done