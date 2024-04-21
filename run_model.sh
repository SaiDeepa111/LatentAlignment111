#!/bin/bash

# python main.py -i 1 -n 100 -m test -s test -l False -c 0 -p save/ -f outputs/log.txt -a 7 -e 2 -o all -r 0.001 -x True
python main_copy.py -i 5 -n 1500 -m train -s train -l False -c 0 -p save/baseline_one/ -f outputs/log.txt -a 8 -e 2 -o one -r 0.001 -x True
python main_copy.py -i 5 -n 1500 -m train -s train -l False -c 0 -p save/baseline_all/ -f outputs/log.txt -a 8 -e 2 -o all -r 0.001 -x True

# -i for number of iterations
# -n for number of samples in use
# -m for the mode of "train" or "test"
# -s for the set  of "train", "test" or "valid"
# -l for using the stored models or not (-l True)
# -c to specify the gpu number
# -p to specify the main folder for the experiment ( Save and load)
# -f to specify the txt file path and name for saving log
# -a for architecture number (7,8,9)
# -e specifying the embedding type ( 1 for bert, 2 for flair, 3 for xlnet)
# -o For specifying the loss mode ("one" for objective 1 and "all" for objective 2)
# -r for specifying the learning rate
# -x for enabling or desabling the modified max pooling
