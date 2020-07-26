# TDDPG-Rec
The code to reproduce the experimental results for "A Text-based Deep Reinforcement Learning Framework for Interactive Recommendation" (In the 24th European Conference on Artificial Intelligence, ECAI 2020).

## Datasets
The data pre-processing codes is also included. You could download Amazon data from *[here](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles)*, and run the amazon.py.

## Runtime Environment
The code has been tested under Windows 10(version 1909) and Ubuntu 16.04 with TensorFlow 1.14.0 and Python 3.6.8.

Support independent training with CPU and joint training with CPU and GPU when CUDA is available.

## Resource
You can download and add these resource to this project under the folder `./resource`.

The pre-trained word vectors is available on *[GloVe.6B](http://nlp.stanford.edu/data/glove.6B.zip)*, which was trained on Wikipedia2014 and Gigaword 5.

The Long Stopword List can be obtained *[here](https://www.ranks.nl/stopwords)*.

## Model Training
Take `Digital_Music` for example. After getting the source data, you should run data process first:

```
python amazon.py 
```

To train our DDPG model on `Digital_Music`: 

```
python DDPG_Rec.py 
```

or our DQN model on `Digital_Music`:

```
python DQN_Rec.py
``` 

You can modify the source codes to run other datasets. For MF-class methods, you should change the input by modify 'method' from 'glove' to 'mf'.
