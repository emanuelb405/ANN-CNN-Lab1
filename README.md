General explanation:
The folders (except the 'Images' folder) contain the models including the
code for training them and the .sh files for starting them on the server 
cluster.
Before you can use them you need to run the 'train_test_split.py' and copy 
resulting train and test folders into the folder that you want to run.
Some folders have an extra file for generating a model which has to happen
on a local machine, since it relies on pretrained keras models.

A more thorough explanation of all models including the solutions can be 
found in the report.

The 'plots_etc.ipynb' contains all the post processing.