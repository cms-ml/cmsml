import tensorflow as tf
from pathlib import Path


def load_model(dir, serving_key='serving_default'):
    dir = Path(dir)
    loaded = tf.saved_model.load(str(dir))
    inference_func = loaded.signatures[serving_key]
    # there is a bug for TFV < 2.8 where python garbage collector kills the model (weak link)
    inference_func._backref_to_saved_model = loaded
    return inference_func



p = '/afs/desy.de/user/w/wiedersb/CMSSW_12_4_0/src/PerfTests/aot_convert/saved_model_12_128'
model = load_model(p, 'batch_size_1')

from IPython import embed; embed()


