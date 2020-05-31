# Training

For an example usage run:

```
./run.sh
```

For an overview of possibilities run `python train_models.py -h`.
(Note: Running the pipeline requires ~10gb of free RAM for cross-subject classification)

# Setup

Models and corresponding figures are saved in the Models/Directory.

The 'Data' directory needs to contain the Folders 'Cross', 'Train'.
The folder 'Cross' additionaly neets to contain a folder 'CV', which contains the following files selected from  train and test data from 'Train':

['rest_105923_4.h5', 'task_motor_105923_1.h5', 'task_working_memory_105923_1.h5', 'task_motor_105923_4.h5', 'task_motor_105923_2.h5', 'task_working_memory_105923_2.h5', 'task_story_math_105923_3.h5', 'rest_105923_1.h5', 'task_story_math_105923_2.h5', 'task_motor_105923_3.h5', 'rest_105923_3.h5', 'task_working_memory_105923_3.h5', 'task_working_memory_105923_4.h5', 'task_story_math_105923_1.h5', 'rest_105923_2.h5', 'task_story_math_105923_4.h5']
