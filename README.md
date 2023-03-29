# AI612 project2 (2023 Spring)

This repository is for the training framework to support AI612 Project 2, Multi-task Multi-source Learning.

## Objective
Given ICU records of 12 hours since the ICU admission, perform 28 prediction tasks, for three datasets. The tasks are:
* Mortality prediction (short, long)
* Readmission prediction
* Diagnosis prediction (17 different diagnoses)
* Length of stay prediction (short, long)
* Final acuity (6 different locations)
* Imminent discharge (6 different locations)
* Creatinine level (5 different levels)
* Bilirubin level (5 different levels)
* Platelet level (5 different levels)
* White blood cell level (3 different levels)

For more information about the Project 2 such as dataset download, please refer to the description docx that we have shared with you.

# Training Framework

### Data structure
When you unzip the compressed data file that we provided you, the file structure will be like this:
```
~/train
├─ mimiciii
│  ├─ PATIENTS.csv
│  ├─ ADMISSIONS.csv
│  ├─ ICUSTAYS.csv
│  ├─ LABEVENTS.csv
│  └─ ...
│
├─ mimiciv
│  ├─ patients.csv
│  ├─ admissions.csv
│  ├─ icustays.csv
│  ├─ labevents.csv
│  └─ ...
│
├─ eicu
│  ├─ patient.csv
│  ├─ lab.csv
│  ├─ medication.csv
│  ├─ infusionDrug.csv
│  └─ ...
│
└─ labels
   ├─ mimciii_labels.csv
   ├─ mimiciv_labels.csv
   └─ eicu_labels.csv
```
* csv files in each EHR (mimiciii, mimiciv and eicu) are the EHR tables where ICU ids (`ICUSTAY_ID`, `stay_id`, `patientunitstayid`) are anonymized and events are truncated within 12 hours since the ICU admission.
* Each label csv file (mimiciii_labels.csv, mimiciv_labels.csv, eicu_labels.csv) provides labels of 28 tasks in the form of string describing python list for each corresponding ICU id. 

### What you need to implement are as follows:
* [preprocess/00000000_preprocess.py](preprocess/00000000_preprocess.py)
    * Feature preprocessing function
    * This is where you use your creativity to handle heterogeneous EHR formats
    * Input
        * Path to the three datasets (typically, '~/train/')
        * Destination directory path to dump your processed features
    * Output
        * Dumped input features to your model
    * Run command
        ```shell script
        $ python preprocess/00000000_preprocess.py ~/train/ --dest output/
        ```
    * Notes
        * This script should dump processed features to the `--dest` directory
        * Note that `--dest` directory will be an input to your dataset class (i.e., --data_path)
        * You can dump any type of files such as json, cPickle, or whatever your dataset can handle
        
* [data/00000000_dataset.py](data/00000000_dataset.py)
    * You need to implement your own dataset class extending torch.utils.data.Dataset
    * Input
        * Path to the output of the preprocessing code (should match with --dest in the preprocessing code)
    * Output
        * Samples to be used as an input to your own model as well as labels for them
    * Notes
        * You must return a dictionary (not a tuple) from `__getitem__` or `collator` so that the data loader yields samples in the form of python dictionary.
        * Example
            ```python
            class MyDataset(...):
                ...
                def __getitem__(self, index):
                    (...)
                    return {"data": data, "label": label} # should be a dictionary, not a tuple
            ```
* [models/00000000_model.py](models/00000000_model.py)
    * You can create your own model to handle heterogeneous EHR formats
    * Input
        * Output of the data loader in the form of **keyword arguments** (i.e., `**samples`) where each key is corresponded with the output dictinoary of the dataset that you implemented.
    * Output
        * Logits in the shape of (batch, 52)
    * Notes
        * You should implement some utility functions in your model:
            * `get_logits(cls, net_output)` returns the logits from the net's output
                * Assure that `get_logits` return the logits in the shape of (batch, 52)
            * `get_targets(cls, sample)` returns the targets (gt labels) from the sample (dictionary)
                * Assure that `get_targets` return the ground truth labels in the shape of (batch, 28)

More details about each implementation are described in the corresponding python file.

### To conduct the experiments, run:
```shell script
$ python train.py --student_number $student_number --data_path $data_path
```
$student_number should be set to your student number, which should match with your dataset and model name.  
$data_path should be set to the path to the output of the preprocessing code.  
If you want to control some hyperparameters such as learning rate or batch size, add command line parameters for those.

After you run the experiments, the framework will automatically make directories and output the model checkpoints at the latest epoch for every `--save_interval` as well as other stuffs such as training logs or configurations.
```shell script
~/ai612-project-2023
├─ ...
└─ outputs
   └─ $date
       └─ $time
            ├─ .config
            │   └─ config.yaml
            ├─ $save_dir
            │   ├─ checkpoint_last.pt
            │   └─ checkpoint{%d}.pt
            └─ train.log
```
$date and $time are describing the time that you run the codes  
$save_dir is the same with `--save_dir` (default: `checkpoints`)

When you submit the model parameters, please submit one of these auto-generated checkpoints after renaming with your student id (see below for what to submit).

### You can modify the framework wherever you want, but there are some constraints you have to be aware of:
* Please keep in mind that we will test your model with only the two command line arguments: `--student_number` and `--data_path`, which means that we may not be able to include your own command line arguments if you added.
* Also, please note that you only submit the implementations of preprocessing, dataset and models, so your modifications on other source codes such as `train.py` or `trainer.py` will not be included when we test your model.
* Hence, if you want to add command line arguments to tune your model such as model hyperparameters like dropout probability, we recommend you to add whatever you want in the **training phase**, but set the default values for those variables to the final values **when you submit** so that your code does not need any explicit arguments other than `--student_number` and `--data_path`.


### What you need to submit are as follows:
* `requirements.txt`
    * You can make this file for dependencies using this command
        ```shell script
        $ pip freeze > requirements.txt`
        ```
* `{student_id}_preprocess.py`
    * Your implementation codes for feature pre-processing
* `{student_id}_dataset.py`
    * Your implementation codes for dataset
* `{student_id}_model.py`
    * Your implementation codes for model
* `{student_id}_checkpoint.pt`
    * The auto-generated checkpoint

# Features
We provide basic functionalities to support your training.
### Control parameters for training
* `--lr`: learning rate
* `--batch_size`: batch size
* `--max_epoch`: max training epoch
* `--valid_percent`: percentage for validation subset
* `--num_workers`: num workers for loading batch
* `--save_interval`: save a checkpoint eveny N epochs

### Distributed training
* We support distributed data parallel training in our framework.
* If you want to run the experiments with multiple GPUs, set the `--distributed_world_size` as the number of GPUs that you want to distribute (should be less than the total number of GPUs in you station)

### Logging with WandB
* We also provide logging with WandB.
* If you want to use [Weight and Biases](https://wandb.ai/site) for logging, please explicitly pass the following arguments:
    * `--wandb_entity`: WandB entity(team) name to use for logging
    * `--wandb_project`: WandB project name to use for logging
    * If you want to specify the run name (default: 'checkpoint'), prepend environment variable like `WANDB_NAME=your_wandb_run_name python train.py ...`
* It may request you to log in to WandB

### Optimizer
* The default optimizer is set to the [Adam](https://arxiv.org/abs/1711.05101) optimizer.
    * The hyperparameters that you can tune on this module are as follows:
        * `--adam_betas`: betas for Adam optimizer
        * `--adam_eps`: epsilon for Adam optimizer
        * `--weight_decay`: weight decay
* You can define your own optimizer by implementing an additional optimizer class into the directory: ~/optim/, which should extend `optim.Optimizer` class. Then, run the training code with `--optimizer=your_optimizer_name`.

### LR scheduler
* The default lr scheduler is set to a fixed scheduler supporting lr decaying & warm-up.
    * The hyperparameters that you can tune on this module are as follows:
        * `--force_anneal`: force annealing at specific epoch
        * `--lr_shrink`: shrink factor for annealing, lr_new = (lr * lr_shrink)
        * `--warmup_updates`: warmup the learning rate linearly for the first N updates
* You can define your own lr scheduler by implementing an additional lr scheduler class into the directory: ~/optim.lr_scheduler/, which should extend `optim.lr_scheduler.LRScheduler`. Then, run the training code with `--lr_scheduler=your_lr_scheduler_name`.

Find more arguments from `train.py` where the details such as data types and descriptions are also provided. We encourage you to try whatever you want!

# Expected Errors
We curate expected errors and how to solve it in this section. If you cannot find out your error here, please report it to the issue so that we can curate them.
```shell script
"ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'"
```
* This error usually occurs when the validation subsets are so small that some gt classes are missed out. If this error happens, you need to consider increasing the validation size by `--valid_percent`.
* Or you can try another random seed to find out perfect validation subset where all the gt classes are survived.