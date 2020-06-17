#!/usr/bin/env bash

python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=10 training.use_original_labels=True training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 model.initialize=True training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 model.initialize=False training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 model.initialize=True training.reinitialize=False training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 training.optimizer.name=adam training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=100 training.n_clusters=50
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=50 training.reassign=4 training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=50 training.reassign=4 training.n_clusters=100
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=50 training.reassign=20 training.n_clusters=10
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=64 \
    training.num_epochs=50 training.reassign=20 training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=32 \
    training.num_epochs=100 model.name=ResNet50 training.n_clusters=20
python -m src.deep_clusterers.train dataset=voc07 training.batch_size=32 \
    training.num_epochs=100 model.name=ResNet152 training.n_clusters=20
