# Data Simulation and Generation

## BACKGROUND:

Self-organizing maps (SOM) are a machine learning technique that can help discover structure and order in a high-dimensional dataset, often very high-dimensional.  It uses a simple technique to map data from a high-dimensional space (which can be the original data measurements, if they are appropriately scaled), down to a 1, 2, or 3 dimensional decision space for visualization and easier understanding and division/classification of the space.

I have already used an open-source implementation of the SOM algorithms, with unclear results.  The literature is unclear what types of structure will be preserved and revealed from the original data, and what types of preprocessing/dimension reduction may improve results.


## PURPOSE:

The purpose of this program is to generate simulated data with various confounds to see best how to recover the original signal using the SOM tool and different types of preprocessing.


## REQUIREMENTS:

1. Generate data to be fed to another python module for SOM analysis.
	a. Generate data as an numpy array of floats.
	b. Create a csv file of a single data array for storage and use.

2. Confounds/variations on the data
	a. Start with clean, linear and linearly-spaced data.
	b. Allow linear data to be unevenly spaced.
	c. Randomly translate data points radially from linear axis.
	d. Rotate linear-ish dataset in 3D space. 
	e. Translate entire linear-ish dataset uniformly arbitrarily in 3D space.
	f.  Add random noise to extra dimensions to be stacked with this 3D dataset
	g. (Alternative to e and f.) Translate entire linear-ish dataset uniformly arbitrarily in the higher-
	    dimensional space.
		I. Add noise to high-dimensional space in some meaningful way that does not destroy
                        the coherent, topological arrangement from the original data.

3. Generate the data in a slightly more complex format, which tags each data point with a tag for the point’s location in the original, linear dataset.  This allows us to more easily see how the data topology is retained in the final, SOM-processed dataset from the original linear dataset.


## INSTALLATION INSTRUCTIONS:

You need a Python 3 environment (built in Python 3.4.3 on a Mac using the Framework Python because of matplotlib issues).

Python package dependencies:
numpy 1.10.1
matplotlib 1.5.0

Can somewhat exercise the code with the enclosed unit tests if the above environment is configured properly, using the command python (or python3, or whatever command it takes to invoke your Python 3):

`python test_data_sim.py`

This invokes the unittest framework in an unusually interactive mode, because for most of the (currently 10) unit test, it brings up a matplotlib plot showing the results of the unit operation.


Of the requirements, 1a, 2a, 2c, 2d, 2e, and 2f have already been implemented and tested.  1b will be helpful for documentation, 2b, 2g and 2gI are interesting variations to test, and 3 makes testing the SOM much easier.


## TO DO:

Create a data structure to include a label with each point, so that it can be identified, see whether it is categorized correctly, and see how severe the error is.

Add the ability to jitter the initial 1-D sequence, so that they are not evenly spaced.

Add the ability to generate two differently linear, fogged datasets, and and them together as another type of confound.

Add the ability to save the generated data as a csv file.

Add the ability to create higher-dimensional datasets in different ways, as 2g and 2gI.
