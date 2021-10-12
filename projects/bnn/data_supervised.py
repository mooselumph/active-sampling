
import tensorflow_datasets as tfds
import tensorflow as tf
import jax.numpy as jnp


import numpy as np
import matplotlib.pyplot as plt

from utils import make_absolute


class data_generator:
    def __init__(self, 
            path,
        ):

        
    def __call__(self,test_split=False,return_info=False):

        
        

def get_stead(hdf_file=STEAD_HDF5,csv_file=STEAD_CSV, batch_size=30, **kwargs):

    
    gen = stead_generator(hdf_file,csv_file,**kwargs)

    train_ds = tf.data.Dataset.from_generator(
            gen,
            (tf.float32,tf.int32),
            (tf.TensorShape((num_samples,num_channels)),tf.TensorShape((num_samples))),
        )

    test_ds = tf.data.Dataset.from_generator(
            gen,
            (tf.float32,tf.int32),
            (tf.TensorShape((num_samples,num_channels)),tf.TensorShape((num_samples))),
            args = [True],
        )

    train_ds = tfds.as_numpy(train_ds.batch(batch_size))
    test_ds = tfds.as_numpy(test_ds.batch(batch_size))

    return train_ds, test_ds