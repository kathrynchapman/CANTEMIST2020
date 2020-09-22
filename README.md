# CANTEMIST2020
This is the (my) repo for the CANTEMIST 2020 Shared Task - coding

To run the classifiers, one must first clone the official metric repo: https://github.com/TeMU-BSC/cantemist-evaluation-library
```
git clone https://github.com/TeMU-BSC/cantemist-evaluation-library.git
```
Then, from within the CANTEMIST2020 directory: <br>
-run the label attention classifier with:
```
python run_classifier.py 
      --data_dir processed_data/cantemist 
      --model_type bert 
      --model_name_or_path bert-base-multilingual-cased 
      --output_dir [whatever you want] 
      --doc_max_seq_length 200
      --label_max_seq_length 11
      --loss_fct bbce
      --num_train_epochs 45
      --doc_batching

 ```
