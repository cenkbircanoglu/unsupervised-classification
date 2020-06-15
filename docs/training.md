



### Training 

```bash
python -m src.deep_clusterers.train
```


#### Example Override
```bash
python -m src.deep_clusterers.train dataset.sample_size=2000
```


```bash
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.optimizer.name=adam training.reassign=1 training.optimizer.lr=1e-3
```

python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=10 training.assign_real_labels=True
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=1
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=1
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=1 model.initialize=True
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=1 model.initialize=False
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=1 training.optimizer.name=adam
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=1 training.optimizer.name=adam
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=1 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=1 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=1 training.n_clusters=50
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=1 training.n_clusters=50
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=True training.reassign=4 training.n_clusters=20
python -m src.deep_clusterers.train dataset.sample_size=70000 training.batch_size=1024 training.num_epochs=1000 training.assign=False training.reassign=4 training.n_clusters=20