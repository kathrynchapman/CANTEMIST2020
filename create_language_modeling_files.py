import argparse
import os
import pickle
# from nltk.tokenize import sent_tokenize
from transformers import XLMRobertaTokenizer

cantemist_path = "cantemist/"
out_path = "processed_data/"
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(out_path + cantemist_path):
    os.mkdir(out_path + cantemist_path)


def save(fname, data):
    with open(fname, "wb") as wf:
        pickle.dump(data, wf)


class LanguageModelingDataReader():
    def __init__(self, args=''):
        self.args = args
        self.train_path = os.path.join(cantemist_path, "train-set/")
        self.test_path = os.path.join(cantemist_path, "test-set/")
        self.background_path = os.path.join(cantemist_path, "background-set/")
        self.dev_path = os.path.join(cantemist_path, "dev-set1/")
        self.data_list = []
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.model_name_or_path)

    def construct_data_dict(self, train=False, test=False, background=False):
        """
        Constructs the dictionary containing all data
        self.data_dict['doc_id'] = 'text'
        :param train_or_test: str: 'train', 'test'; for finding the proper directory
        :return: None
        """
        if train:
            path = self.train_path + 'cantemist-coding/txt/'
        elif test:
            path = self.test_path + 'cantemist-ner/'
        elif background:
            path = self.background_path
        else:
            path = self.dev_path + 'cantemist-coding/txt/'

        all_file_names = os.listdir(path)
        for f_name in all_file_names:
            if f_name[-3:] == 'txt':
                with open(path + f_name) as f:
                    text = f.read().replace('\n', ' ').replace('\t', ' ')

                    tokens = self.tokenizer.tokenize(text)
                    while tokens:
                        self.data_list.append(self.tokenizer.convert_tokens_to_string(tokens[:self.args.msl]))
                        tokens = tokens[self.args.msl - self.args.stride:]
        self.data_list = [self.tokenizer.convert_tokens_to_string(t) for t in self.data_list]

    def write_data(self):
        with open(
                out_path + cantemist_path + 'LM_{}_{}_{}.txt'.format(self.args.model_name_or_path, str(self.args.msl),
                                                                     str(self.args.stride)),
                'w') as f:
            for line in self.data_list:
                f.write(line + '\n')


if __name__ == '__main__':
    """
    Namespace(adam_epsilon=1e-08, block_size=150, cache_dir=None, config_name=None, 
    device=device(type='cuda'), do_eval=False, do_train=True, eval_all_checkpoints=False, 
    eval_data_file=None, evaluate_during_training=False, fp16=False, fp16_opt_level='O1', 
    gradient_accumulation_steps=1, learning_rate=5e-05, line_by_line=True, local_rank=-1, 
    logging_steps=500, max_grad_norm=1.0, max_steps=-1, mlm=True, mlm_probability=0.15, 
    model_name_or_path='xlm-roberta-base', model_type='xlmroberta', msl=150, n_gpu=4, 
    no_cuda=False, num_train_epochs=3.0, output_dir='futher_pretrained_xlmr_base-0/', 
    overwrite_cache=False, overwrite_output_dir=False, per_gpu_eval_batch_size=4, per_gpu_train_batch_size=4, save_steps=500, save_total_limit=None, seed=42, server_ip='', server_port='', should_continue=False, stride=75, tokenizer_name=None, train_batch_size=16, train_data_file='processed_data/cantemist/LM_xlm-roberta-base_150_75.txt', warmup_steps=0, weight_decay=0.0)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help="Whether to redo all of the pre-processing.")
    parser.add_argument('--model_name_or_path', default='xlm-roberta-base', type=str,
                        help="What type of BERT flavor we're working with")
    parser.add_argument('--msl', type=int, default=150, help="Intended max seq len")
    parser.add_argument('--stride', type=int, default=75, help="Overlap between documents")

    args = parser.parse_args()

    reader = LanguageModelingDataReader(args)
    reader.construct_data_dict(train=True)
    reader.construct_data_dict()
    reader.construct_data_dict(test=True)
    reader.construct_data_dict(background=True)
    reader.write_data()
