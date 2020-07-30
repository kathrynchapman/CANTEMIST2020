# CANTEMIST2020
This is the (my) repo for the CANTEMIST 2020 Shared Task - coding

To run the classifiers, one must first clone the official metric repo: https://github.com/TeMU-BSC/cantemist-evaluation-library
```
git clone https://github.com/TeMU-BSC/cantemist-evaluation-library.git
```
Then, from within the CANTEMIST2020 directory: <br>
-run the baseline classifier with:
```
python run_baseline_classifier.py 
      --data_dir processed_data/cantemist 
      --model_type bert 
      --model_name_or_path bert-base-multilingual-cased 
      --output_dir [whatever you want] 
      --doc_max_seq_length 256
      --loss_fct bce
      --num_train_epochs 45
      --doc_batching
      --do_normal_class_weights 
      --do_ranking_loss
 ```


-run the label attention classifier with:
```
python run_label_attn_classifier.py 
      --data_dir processed_data/cantemist 
      --model_type bert 
      --model_name_or_path bert-base-multilingual-cased 
      --output_dir [whatever you want] 
      --doc_max_seq_length 256
      --label_max_seq_length 15
      --loss_fct bce
      --num_train_epochs 45
      --doc_batching
      --do_normal_class_weights 
      --do_ranking_loss
 ```
This is still a work in progress (obv)
