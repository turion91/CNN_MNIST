# CNN_MNIST
Modular CNN for both fashion and digit MNIST dataset. Possibility to go up to 4 conv-pool and 4 dense layers.
The repository is composed of four files: CNN_fashion_main.py, CNN_layers.py, Dense_layers.py, variables.py. CNN_layers contains helper functions
to call either one to four conv/pool layers, Dense_layers contains helper functions to call either one to four dense layers. CNN_fashion_main.py contains the
main script, this will build and run the CNN to classify images (So far only test on both fashion and digit MNIST dataset). Finally
variables.py contains all variables called by the three other scripts. This should be the only file to change as it will allow the user
to change the image pixel size (default is 28*28) presence or absence of pool layer, number of neurons, conv/pool layers dimensions, presence or 
abscence of dropout.
