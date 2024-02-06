NeuralNetwork.py contains 
	- all the custom pytorch modules used to create both the residual MLP and residual networks
	for both, the general patten is create a linear module, 
	use it to create a "block", 
	and finally use both of these to create a network
	
NN_utils.py contains 
	- the pytorch lightning module used to define the trainer
	- Recorder class used to keep track of validation performances
	- Custum pytorch loss functions
	- helper function to create the dataloader

experiments_tabular.py contains script for "Small Tabular Datasets" experiment
make_figure1.py contains script to create Figure 1 from recorded data
experiments_deep.py contains script for "Ablation Study on Deep Networks"