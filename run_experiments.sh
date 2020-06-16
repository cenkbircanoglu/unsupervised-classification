#!/usr/bin/env bash


python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=10 training.use_original_labels=True
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=50 training.reassign=1 model.initialize=True
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=50 training.reassign=1 model.initialize=False
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=50 training.reassign=1 training.optimizer.name=adam
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=50 training.reassign=1 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=50 training.reassign=1 training.n_clusters=50
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=25 training.reassign=4 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=25 training.reassign=4 training.n_clusters=100
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=10 training.reassign=20 training.n_clusters=10
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 \
    training.num_epochs=10 training.reassign=20 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=128 \
    training.num_epochs=50 training.reassign=1 model.name=ResNet50
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=128 \
    training.num_epochs=50 training.reassign=1 model.name=ResNet152



python -m src.deep_clusterers.train dataset.sample_size=100000 training.batch_size=256 \
    training.num_epochs=250 training.reassign=1 model.initialize=False dataset.image_root_folder=./datasets/cifar100/images \
    dataset.groundtruth_label_file=./datasets/cifar100/cls_labels.npy dataset.label_file=./datasets/cifar100/cls_labels.npy \
    dataset.name=cifar100 training.n_clusters=100