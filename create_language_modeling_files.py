import argparse
import os
import pickle
from nltk.tokenize import sent_tokenize
from transformers import XLMRobertaTokenizer


cantemist_path = "cantemist/"
out_path = "processed_data/"
if not os.path.exists(out_path):
    os.mkdir(out_path)


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
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.model_name)

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
            id_list = self.test_ids
        else:
            path = self.dev_path + 'cantemist-coding/txt/'

        all_file_names = os.listdir(path)
        counter = 0
        for f_name in all_file_names:
            with open(path + f_name) as f:
                text = f.read().replace('\n', ' ').replace('\t', ' ')
                sentences = sent_tokenize(text)


                tokens = self.tokenizer.tokenize(text)
                while tokens:
                    self.data_list.append(self.tokenizer.convert_tokens_to_string(tokens[:512]))
                    tokens = tokens[512:]
                # text = sent_tokenize(text)
                # tokenized_sents = [self.tokenizer.tokenize(t) for t in text]
                #
                # if text:
                #     self.data_list += text
        # print(counter)
        for d in self.data_list:
            print(d)
        # print(self.data_list)
        print(len(self.data_list))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', help="Whether to redo all of the pre-processing.")
    parser.add_argument('--model_name', type=str, help="What type of BERT flavor we're working with")

    args = parser.parse_args()

    reader = LanguageModelingDataReader(args)
    reader.construct_data_dict(train=True)