# STREAK Example Code

Implementation of the STREAK algorithm for streaming maximization of weakly submodular functions.

- Ethan R. Elenberg, Alexandros G. Dimakis, Moran Feldman, and Amin Karbasi. ‘‘Streaming Weak Submodularity: Interpreting Neural Networks on the Fly’’, to appear in *Proc. Neural Information Processing Systems (NIPS)*, 2017. 
[arXiv (preprint)](http://arxiv.org/abs/1703.02647)

### Requirements

- Directory ‘retrained' that contains the black box models
	
	-- bottleneck\_fc\_model.h5 (keras)

	-- classify\_image\_graph\_def.pb and output\_labels.txt (tensorflow)
	 
- Directory ‘sunflowers’ that contains jpeg images from class sunflowers to use as queries

- Directory ‘daisy' that contains jpeg images from class daisy to use as queries

- Directory ‘outputs’ to save the output images

- LIME, TensorFlow and/or Keras, NumPy, and scikit-image packages

### Usage

 The main scripts are [streakRegressionExample.py](./blob/master/streakRegressionExample.py) and [streakInterpretationExample.py](./blob/master/streakInterpretationExample.py). The Jupyter notebook [StreakImageRetraining.ipynb](./blob/master/StreakImageRetraining.ipynb) is also available as a convenient walkthrough of streakIntrepretationExample. [tf_predict.py](./blob/master/tf_predict.py) can also be used from the command line to load the tensorflow model and predict labels for a list of images.

```sh
python streakRegressionExample.py
python streakInterpretationExample.py image1.jpg image2.jpg
python tf_predict.py image1.jpg image2.jpg
```
