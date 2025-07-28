# Getting Started

## Environment

Python 3.7.13, PyTorch 1.10.0, scikit-learn 1.0.2, fastText 0.9.2, and DGL 0.9.2 are suggested.
这块的环境请看项目根目录下的requirements.txt文件，一步到位。

## Dataset

D1: https://github.com/CloudWise-OpenSource/GAIA-DataSet

D2： AIOps22

D3： OnlineBoutique

D4： TrainTickert

## Demo

We provide a demo. Please run:

```
python main.py --config gaia_config.yaml
```

## Parameter Description in the Demo

### fastText \& Instance Embedding

* `vector_dim`: The dimension of event embedding vectors. (default: 100)
* `sample_count`: The number of samples per type after data augmentation. (default: 1000)
* `edit_count`: The number of events modified per sample during data augmentation. (default: 1)
* `minCount`: The minimum number of occurrences of the event (events that occur less than this number are ignored). (default: 1)

### DGL

* `epoch`: Training rounds. (default: 6000)
* `batch_size`: The number of samples contained in a batch of data. (default: 1000)
* `win_size`: The length of the judgment window for ending training early. (default: 10)
* `win_threshole`: The thresh for ending training early. (default: 0.0001)
* `lr`: The learning rate. (default: 0.001)
