# BSSL
Binary-Classifiers-Enabled Filters for Semi-Supervised Learning
main.py contain the main code for training the models. 

to train for proposed approach, with 10 exmaples per class, and with augmentation dataset ESC10  and model large (13 layers)
use below command

CUDA_VISIBLE_DEVICES=0 python main.py --learning SSL --examples 10 --augmentation 1 --dataset ESC10 --model large

learning: SS-> supervised leanring, PSSL-> semi-supervised learning with pseudo labeling, SSL-> binary classifiers semi-supervised learning 

examples : Examples per class 

Augmentation: Allowed(1) or not allowed (0)

Dataset: Name of dataset 

model : large (we used for paper), for faster training, you can try small model 


For below files, binary classifiers should be trained 

1. Cascade.py 
2. NonCascade.py
3. RankBinaryClassifier.py


ExampleComparison.py saves the predicted labels by SSL and PSSL, and save them. Further you can visualize them using t-sne. 

For correct version of libraries, check run.sh 
