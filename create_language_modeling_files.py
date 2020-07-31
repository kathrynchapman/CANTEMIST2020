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
        self.train_path = os.path.join(cantemist_path, "train-set-to-publish/")
        self.test_path = os.path.join(cantemist_path, "test-background-set-to-publish/")
        self.dev_path = os.path.join(cantemist_path, "dev-set1-to-publish/")
        self.data_list = []
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.model_name_or_path)

    def construct_data_dict(self, train=False, test=False):
        """
        Constructs the dictionary containing all data
        self.data_dict['doc_id'] = 'text'
        :param train_or_test: str: 'train', 'test'; for finding the proper directory
        :return: None
        """
        if train:
            path = self.train_path + 'cantemist-coding/txt/'
        elif test:
            path = self.test_path
        else:
            path = self.dev_path + 'cantemist-coding/txt/'

        all_file_names = os.listdir(path)
        for f_name in all_file_names:
            with open(path + f_name) as f:
                text = f.read().replace('\n', ' ').replace('\t', ' ')

                tokens = self.tokenizer.tokenize(text)
                while tokens:
                    self.data_list.append(self.tokenizer.convert_tokens_to_string(tokens[:self.args.msl]))
                    tokens = tokens[self.args.msl - self.args.stride:]
        self.data_list = [self.tokenizer.convert_tokens_to_string(t) for t in self.data_list]

    def write_data(self):
        with open(
            out_path + cantemist_path + 'LM_{}_{}_{}.txt'.format(self.args.model_name_or_path, str(self.args.msl), str(self.args.stride)),
            'w') as f:
            for line in self.data_list:
                f.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help="Whether to redo all of the pre-processing.")
    parser.add_argument('--model_name_or_path', type=str, help="What type of BERT flavor we're working with")
    parser.add_argument('--msl', type=int, default=512, help="Intended max seq len")
    parser.add_argument('--stride', type=int, default=75, help="Overlap between documents")

    args = parser.parse_args()

    reader = LanguageModelingDataReader(args)
    reader.construct_data_dict(train=True)
    reader.construct_data_dict()
    reader.construct_data_dict(test=True)
    reader.write_data()
