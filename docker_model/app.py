"""
Purpose: Serve trained peak-finder model via a flask app.

Typical call:

```
python app.py --port=5000
```

Where --port is the port that will be used to expose the app
"""

import tensorflow as tf

import numpy as np
import numpy.random as npr

import json

import flask as fk

import argparse


#########

app = fk.Flask(__name__)
px_count = 1024
    
#####

@app.route("/findpeak/<src_signal>")
def findpeak(src_signal):
    """
    Given a list of input signal values, return the inference from the model, that shows where the peaks would
    be. Input and output is done in a raw JSON format.
    
    Further imporovements would be to serialize it with protobuf and CRC32, or similar
    """

    # extact the input signal into a numpy array
    signal_arr = np.squeeze(np.array(json.loads(src_signal)))

    # convert to tensorflow array, feed into the model, and get output
    signal_arr_tf = tf.constant(signal_arr, dtype=tf.float32)
    peak_arr = loaded_model(tf.reshape(signal_arr_tf, shape=(1, -1, 1)))

    # encode output as a json-compatible array
    peak_str = '[' + ','.join([f'{val:.3f}' for val in np.squeeze(peak_arr.numpy())]) + ']'

    # return as unicode
    return f"{peak_str}".encode('utf8')
    
######

if __name__ == '__main__':
    # sort out the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="port for the flask app", required=True, type=int)
    args = parser.parse_args()

    # load the TF model
    loaded_model = tf.saved_model.load('peakfinder_final_model')
    
    # will launch the run
    app.run(port=args.port, host='127.0.0.1')
