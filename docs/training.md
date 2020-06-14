



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