# CANTEMIST2020
This is the (my) repo for the CANTEMIST 2020 Shared Task - coding

To run the classifiers, one must first clone the official metric repo: https://github.com/TeMU-BSC/cantemist-evaluation-library
```
git clone https://github.com/TeMU-BSC/cantemist-evaluation-library.git
```
Then, from within the CANTEMIST2020 directory: <br>
-run the baseline classifier with:
```
python run_label_attn_classifier.py 
      --data_dir processed_data/cantemist 
      --model_type bert 
      --model_name_or_path bert-base-multilingual-cased 
      --output_dir [whatever you want] 
      --doc_max_seq_length 256
      --label_max_seq_length 15
 ```
  the following additional flags can be added:
  ```
      --loss_fct [bce, bbce] 
  ```
   'bce': BCEWithLogitsLoss
   'bbce' BalancedBCEWithLogitsLoss
   
```
      --num_train_epochs [int]
```
   I've been using 45
```
      --doc_batching 
```
  whether to using the sliding window approach to fit each document into a mini batch
```
      --do_normal_class_weights 
```
  whether the use the static class weights calculated by (num negative examples)/(num positive examples)
```
    --do_iterative_class_weights
```
  whether to use dynamically calculated class weights, which weights incorrectedly predicted labels higher than correctly predicted labels
```
    --do_ranking_loss
```
  whether to add the ranking loss
