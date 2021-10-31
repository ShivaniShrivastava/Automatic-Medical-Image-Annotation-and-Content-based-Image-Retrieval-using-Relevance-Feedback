Framework required:
	1. Tensorflow

#NOTE:
The dataset folders, label.txt and all the python files are put in a folder called "seperated". All the paths are set relative to that folder.

label.txt : It consists of the names of folders that contains the images.
			The names of folders are same as the names of the classes.
			The line number corresponding to the names in the file are the position of logits in tensorflow.


build_image_data.py : It builds of the tfRecord files by referring label.txt.

dir_traversal_tfrecord.py : Lists the names of all tfRecord in the folder

read_tfrecord_data.py : Lists the contents of tfRecord.

cnn_ir.py : It consists of the main code. It has three functions :
		1. train : It trains the CNN model on the classes mentioned in label.txt
		2. eval : It tests the CNN model and directs the output of the assigned labels corresponding to images to as csv file.
		3. train2 : It considers the relevance feedback of the user and backpropagates the error only on the fully connected layers.

retrieval.py : It reads the csv file obtained and runs performance metrics on it.

pred1.csv : The obtained csv file that consists of the actual class label, name of the image and the assigned class 			label.


The commands for running the code :
1. python build_image_data.py
2. python cnn_ir.py
3. python retrieval.py