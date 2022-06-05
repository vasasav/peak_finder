# peak_finder

Tensorflow based model to detect full width at half-maximum of peaks in noisy data. Includes conterization of the model.

## Data generation, model training, requests

Most of the action is in `peak_finder.ipynb` which includes functions to generate peaks for training. Code to compose the convolutional model as well as to train and save it. 

Note that there is an air-gap between data generation and model training. The training data is saved into an HDF5 file, which is not included in the GIT commit.

In addition to training and validating the model, additional explored tasks include exploration of trained model by optimizing model input in order to obtain the desired output.

Finally, the notebook includes instructions to package the trained model into a docker container, and to make it available for inference via request calls.

## Docker model

Directory `docker_model` includes a simple docker file and a basic `app.py` file to serve trained model, saved in a `SavedModel` format, via a Flask library.

NB! Not committing the actual model (`peakfinder_final_model`). See notebook `peak_finder.ipynb` for save instructions

