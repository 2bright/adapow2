# Adapow2
  Adapow2 is a serial of adaptive gradient descent optimizers by adjusting the power of 2 of a tiny step size.

# usage

## install

```
git clone git@github.com:2bright/adapow2.git
cd adapow2
pip install -e .
```

!!! MUST !!!
  In order to run adapow2, you must replace 'training_arrays.py' of your tensorflow installation with 'adapow2/tf-keras-hack/training_arrays.py'.
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
cd /path/to/adapow2/examples/tests
python3 test_MultiStepProbing.py
```

