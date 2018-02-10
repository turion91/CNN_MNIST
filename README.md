# CNN_MNIST
Modular CNN for both fashion and digit MNIST dataset. Possibility to go up to 4 conv-pool and 4 dense layers.
The repository is composed of four files: CNN_fashion_main.py, CNN_layers.py, Dense_layers.py, variables.py. CNN_layers contains helper functions
to call either one to four conv/pool layers, Dense_layers contains helper functions to call either one to four dense layers. CNN_fashion_main.py contains the
main script, this will build and run the CNN to classify images (So far only test on both fashion and digit MNIST dataset). Finally
variables.py contains all variables called by the three other scripts. This should be the only file to change as it will allow the user
to change the image pixel size (default is 28*28) presence or absence of pool layer, number of neurons, conv/pool layers dimensions, presence or 
abscence of dropout.
There is also the possibility to only work with 2 classes instead of the 10 present in the regular MNIST datasets.
There is the possibility to tune the number of conv layers, the hidden layers, and number of neurons.
Working on both datasets will currently be memory extensive, as well as computation instensive so using tensorflow on a good nvidia GPU is advised (by playing around with the parameters in variables.py, this script can reach an accuracy of 91.72% on the fashion MNIST dataset, but can still be improved).
TODO: Add the possibility of regularisation like L2, change CNN_layers and Dense_layers to be less code heavy, find a way to avoid loading 
the test set in memory.
 
