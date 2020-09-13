# MLProject2: Gravitational Lens Finding Challenge

## About this project
URL: http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html

The aim of this project was to identify gravitational lenses in computer-generated ground based images, through the use of transfer learning and CNNs.
Two methods where used to train the model:
method 1: Reinitialize all batch norm layers before starting to train
method 2: Don't reinitialize the batch norm layers in which case the model has to be kept in training mode when testing.
The two methods are described more in depth in the report

## To run our model
The CNN trainer is accessible through the run.py file. By default the best configuration is set and method 1 is used. Predictions for the test dataset are outputted to a .csv file when the run.py is ran.
For the model to run:
1. The dataset files need to be in .../data folder present in the same directory. Otherwise they would be automatically downloaded when the main is called (roughly 20 GB)
2. Your computer should have a working version of python with the following packages installed in the working environement:
    * numpy
    * torch
    * torchvision
    * pickle
    * itertools
    * argparse
    * importlib
    * csv
    * glob
    * math
    * sampler
    * os
    * astropy
    * six
    * random
    * gzip
    * shutil
3. To test other configurations flags can be declared. For more information on this, type python run.py --help in the terminal

