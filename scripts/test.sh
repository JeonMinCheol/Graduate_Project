ulimit -n 40960

n_class1=62
n_class2=10
n_class3=100
n_clients=20
rounds=101
n_client_epochs=5
frac=0.4
early_stopping=0
is_testing=0

for non_iid in 1
do

for aggregator in cluster fednova 
do

    python -u ../run.py \
    --model_name cnn \
    --data_root ../datasets/ \
    --data cifar10 \
    --aggregator $aggregator \
    --n_clients $n_clients \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --n_class $n_class2 \
    --optim sgd \
    --min_clusters 2 \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu 0.3 \
    --use_multiple_gpu

done

for aggregator in fedprox fednova cluster
do

    python -u ../run.py \
    --model_name cnn \
    --data_root ../datasets/ \
    --data cifar100 \
    --aggregator $aggregator \
    --n_clients $n_clients \
    --frac $frac \
    --rounds $rounds \
    --n_client_epochs $n_client_epochs \
    --n_class $n_class3 \
    --optim sgd \
    --min_clusters 2 \
    --lr 0.01 \
    --port 50052 \
    --momentum 0.9 \
    --early_stopping $early_stopping \
    --use_gpu \
    --is_testing $is_testing \
    --clients_per_rounds 1 \
    --non_iid $non_iid \
    --gpu 2 \
    --mu 0.3 \
    --use_multiple_gpu

done

done
