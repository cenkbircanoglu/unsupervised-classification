#!/usr/bin/env bash

python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=10 training.use_original_labels=True training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 model.initialize=True training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 model.initialize=False training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 model.initialize=True training.reinitialize=False training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 training.optimizer.name=adam training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 training.n_clusters=200
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=500 training.n_clusters=500
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=250 training.reassign=4 training.n_clusters=200
python -m src.deep_clusterers.train dataset=cifar100 training.batch_siz0e=1024 \
    training.num_epochs=250 training.reassign=4 training.n_clusters=1000
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=100 training.reassign=20 training.n_clusters=1000
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=1024 \
    training.num_epochs=100 training.reassign=20 training.n_clusters=200
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=128 \
    training.num_epochs=500 model.name=ResNet50 training.n_clusters=100
python -m src.deep_clusterers.train dataset=cifar100 training.batch_size=128 \
    training.num_epochs=500 model.name=ResNet152 training.n_clusters=100
