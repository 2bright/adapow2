# Adapow2
  Adapow2是一个基于tensorflow.keras开发的神经网络自适应优化算法，目标是完全替代传统的sgd，并逼近adam的优化速度。该算法在执行每一步优化时，方向与sgd相同，但步长通过自适应机制动态调整，使得该算法有接近adam的优化速度。
  
  Adapow2 is a neural network adaptive optimization algorithm based on tensorflow.keras. The goal is to completely replace the traditional sgd and approximate the optimization speed of adam. When the algorithm performs each step of optimization, the direction is the same as sgd, but the step size is dynamically adjusted by the adaptive mechanism, which makes the algorithm have an optimization speed close to adam.
  
# usage

## install

```
git clone git@github.com:2bright/adapow2.git
cd adapow2
pip install -e .
```

***!!! MUST !!!***

  **In order to run adapow2, you must replace 'training_arrays.py' of your tensorflow installation with 'adapow2/tf-keras-hack/training_arrays.py'.**
  For me, the path is '/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training_arrays.py'.
  The following operations may illustrate what you should do.

```
mv /path/to/tensorflow/python/keras/engine/training_arrays.py /path/to/adapow2/tf-keras-hack/training_arrays.py.backup
ln -s /path/to/adapow2/tf-keras-hack/training_arrays.py /path/to/tensorflow/python/keras/engine/training_arrays.py
```

## run examples

```
git clone git@github.com:2bright/kv_prod_union.git
cd kv_prod_union
pip install -e .

cd /path/to/adapow2/examples/adapow2_vs_adam
python3.6 adam_mnist.py
python3.6 adapow2_mnist.py

cd /path/to/adapow2/examples/tests
python3.6 test_MultiStepProbing.py
```
kv_prod_union is for hyperparameter sampling management.
For 'adapow2/examples/tests', the loss-acc figure is stored in data directory.

## for research
You can set config['store_history_state'] hyperparameter of optimizer to be True, and inspect how step size change during training.
Or you can modify test_MultiStepProbing.py file, use OHS to inspect hyperspace slice of optimization path.

