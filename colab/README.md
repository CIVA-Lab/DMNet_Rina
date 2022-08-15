
Relative paths sometimes do not work well in conda, so the only thing you need to do to run the training scripts is to change the relative paths in the `yaml` files that you need to use to the full path. For example, change `../training_codes/models_imagenet` to `/content/DMNet_Rina/training_codes/models_imagenet`.
The same in the corresponding training python script in case there exists any. 

If the training does not work when it is called using `bash`, copy&paste the scripts in the training file directly into the notebook with the correct inputs that are specified in the bash and run it placing the working directory in `training_codes`. 

