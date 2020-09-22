# CANTEMIST2020
This is the repo for the CANTEMIST 2020 Shared Task - coding track

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
In order to completely replicate results, a pre-trained XLM-RoBERTa-Base model will need
to be further pretrained using the MLM objective on all train, dev, and test data.
The command for this is:
```
python run_language_modeling.py 
      --output_dir futher_pretrained_xlmr_base-0/ 
      --model_name_or_path xlm-roberta-base 
      --model_type xlmroberta 
      --train_data_file processed_data/cantemist/LM_xlm-roberta-base_150_75.txt 
      --line_by_line
      --block_size 150 
      --mlm 
      --msl 150 
      --num_train_epochs 3 
      --do_train
```

Then, the ```run_classifier.py``` script can be ran, changing the following:
```
      --model_type bert 
      --model_name_or_path bert-base-multilingual-cased 
```