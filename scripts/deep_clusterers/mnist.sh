#!/usr/bin/env bash

python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=10 training.use_original_labels=True
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 model.initialize=True
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 model.initialize=False
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 model.initialize=True training.reinitialize=False
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 training.optimizer.name=adam
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 training.n_clusters=20
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=100 training.n_clusters=50
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=50 training.reassign=4 training.n_clusters=20
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=50 training.reassign=4 training.n_clusters=100
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=50 training.reassign=20 training.n_clusters=10
python -m src.deep_clusterers.train dataset=mnist training.batch_size=1024 \
    training.num_epochs=50 training.reassign=20 training.n_clusters=20
python -m src.deep_clusterers.train dataset=mnist training.batch_size=128 \
    training.num_epochs=100 model.name=ResNet50
python -m src.deep_clusterers.train dataset=mnist training.batch_size=128 \
    training.num_epochs=100 model.name=ResNet152
