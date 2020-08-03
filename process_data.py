import os
import random
from collections import defaultdict, Counter
import string
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import argparse
import itertools
import numpy as np
from sklearn.preprocessing import normalize
from utils import *
from transformers import XLMRobertaTokenizer

random.seed(30)
cantemist_path = "cantemist/"
out_path = "processed_data/"
if not os.path.exists(out_path):
    os.mkdir(out_path)


def save(fname, data):
    with open(fname, "wb") as wf:
        pickle.dump(data, wf)


class CantemistReader():
    """
    Reads in Spanish data from 2020 CANTEMIST Challenge, writes reformatted version to tsv file, and binarizes labels
    and pickles resulting data set, filtering out labels under user-specified threshold
    """

    def __init__(self, args=''):
        self.train_path = os.path.join(cantemist_path, "train-set-to-publish/")
        self.test_path = os.path.join(cantemist_path, "test-background-set-to-publish/")
        self.dev_path = os.path.join(cantemist_path, "dev-set1-to-publish/")
        self.train_on_all = args.train_on_all
        self.data_dict = dict()
        self.label_dict = defaultdict(list)
        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []
        self.mlb = MultiLabelBinarizer()
        self.args = args
        self.label_desc_dict = defaultdict(str)
        self.span_dict = defaultdict(list)
        self.class_weight_dict = defaultdict(int)
        self.train_file = 'train_{}_{}'.format(self.args.label_threshold, self.args.train_on_all)
        self.dev_file = 'dev_{}_{}'.format(self.args.label_threshold, self.args.train_on_all)
        self.test_file = 'test_{}_{}'.format(self.args.label_threshold, self.args.train_on_all)
        self.n_disc_docs = 0
        self.total_labeled_docs = 0
        if not os.path.exists('processed_data/cantemist/'):
            os.mkdir('processed_data/cantemist/')

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
        for f_name in all_file_names:
            with open(path + f_name) as f:
                self.data_dict[f_name[:-4]] = f.read().replace('\n', ' ').replace('\t', ' ')
                if test:
                    id_list.append(f_name[:-4])

    def construct_label_dict(self, train=False):
        """
        Creates the dictionary containing the labels for all documents
        self.label_dict[doc_id] = ['label1', 'label2', 'label3', ...]
        :param train_or_test: str: 'train', 'test'; for finding the proper directory
        :return: None
        """

        if train:
            path = self.train_path + 'cantemist-coding/train-coding.tsv'
            id_list = self.train_ids
        else:
            path = self.dev_path + 'cantemist-coding/dev1-coding.tsv'
            id_list = self.dev_ids
        with open(path, 'r') as f:
            for line in f.read().split('\n'):
                try:
                    doc_id, label = line.split('\t')
                    if doc_id == 'file':
                        continue
                    if label == '90800/1':
                        label = '9080/1'
                    self.label_dict[doc_id].append(label)
                    if train == True or self.args.train_on_all:
                        self.class_weight_dict[label] += 1
                    if doc_id not in id_list:
                        id_list.append(doc_id)
                except:
                    continue
        assert set(self.label_dict.keys()).issubset(set(self.data_dict.keys()))

    def construct_label_desc_dict(self):
        """
        Constructs the dictionary containing the label descriptions
        self.label_desc_dict['code'] = 'description'
        :return: None
        """
        try:
            with open('Code_Desc_ES/Morfología_7_caracteres.tsv', 'r') as f1, \
                    open('Code_Desc_ES/Morfología_6_caracteres.tsv', 'r') as f2:
                dat = f1.read().split('\n')
                dat += f2.read().split('\n')
                dat = [d.strip().split('\t') for d in dat if d]
                for code, desc_long, desc_short in dat:
                    if code != 'codigo':
                        self.label_desc_dict[code] = desc_long.strip('"')
        except:
            with open('Code_Desc_ES/Morfología_7_caracteres.tsv', 'r') as f1, \
                    open('Code_Desc_ES/Morfología_6_caracteres.tsv', 'r') as f2:
                dat = f1.read().split('\n')
                dat += f2.read().split('\n')
                dat = [d.strip().split('\t') for d in dat if d]
                for code, desc_long, desc_short in dat:
                    if code != 'codigo':
                        self.label_desc_dict[code] = desc_long.strip('"')
        # now to find the nearest label for those lacking textual descriptions... ugh
        for labels in self.label_dict.values():
            for i, label in enumerate(labels):
                if not self.label_desc_dict[label]:
                    label, desc = self.generate_description(label)
                    labels[i] = label
                    self.label_desc_dict[label] = desc

    def generate_description(self, code):
        """
        Generates code descriptions which are not in the downloaded code descriptions file by exploiting similar
        codes and the general hierarchy of  CIE-O-3
        E.g.:
            8000/0	Neoplasia benigna
            8000/1	Neoplasia de benignidad o malignidad incierta
            8000/3	Neoplasia maligna
            8000/31	Neoplasia maligna - grado I, bien diferenciado
            8000/32	Neoplasia maligna - grado II, moderadamente diferenciado
            8000/33	Neoplasia maligna - grado III, pobremente diferenciado
        xxxx/ -> cell type
        xxxx/x -> cell type + behavior (benign, malignant, etc)
        xxxx/xx -> cell type + behavior + grade (well/moderately/poorly/etc-differentiated)

        We have descriptions for all of the cell types + behaviors but not necessarily for all of the grades
        The method takes in a code for which we cannot find a concrete description and then finds the closest code
        for which we have a description, so either cell type + behavior, the appends the appropriate phrase
        corresponding to the grade; descriptions from
        http://www.sld.cu/galerias/pdf/sitios/dne/vol1_morfologia_tumores.pdf
        Additionally, several codes in the data set are appended with /H and it is unclear what this means; after
        eyeballing the data, it seemed most appropriate to just take the most frequent span associated with
        these codes from the NEN subtask
        :param code:
        :return:
        """
        behavior_dict = {"0": ", benigno",
                         "1": ", incierto si es benigno o maligno",
                         "2": ", carcinoma in situ",
                         "3": ", maligno, sitio primario",
                         "6": ", maligno, sitio metastásico",
                         "9": ", maligno, incierto si el sitio es primario o metastásico"
                         }
        grade_dict = {"1": " - grado I, bien diferenciado",
                      "2": " - grado II, moderadamente diferenciado",
                      "3": " - grado III, pobremente diferenciado",
                      "4": " - grado IV, indiferenciado, anaplásico"

                      }
        try:
            assert code[4] == '/'
        except:
            code = code[:4] + code[5:]

        if code[-1] == 'H' or code[-1] == 'P':
            description = self.span_dict[code].most_common(1)[0][0]
        elif self.label_desc_dict[code[:6]]:
            prefix = code[:6]  # 8000/1
            description = self.label_desc_dict[prefix]
            if len(code) > 6:
                description += grade_dict[code[6]]
        else:
            prefix = code[:5]
            suffix = 0
            nearest_code = ''
            while not self.label_desc_dict[nearest_code]:
                nearest_code = prefix + str(suffix)
                suffix += 1
            to_remove = {'maligno', 'benigno', 'maligna', 'benigna'}
            # remove mentions of whether the tumor is benign or malignant
            nearest_desc = ' '.join([w for w in self.label_desc_dict[nearest_code].split() if w not in to_remove])
            description = nearest_desc + behavior_dict[code[5]]
            if len(code) == 7:
                description += grade_dict[code[6]]

        return code, description

    def construct_span_dict(self, train=False):
        if train:
            path = self.train_path + 'cantemist-norm/'
        else:
            path = self.dev_path + 'cantemist-norm/'
        all_file_names = [f for f in os.listdir(path) if f[-4:] == '.ann']
        for f_name in all_file_names:
            with open(path + f_name) as f:
                for line1, line2 in itertools.zip_longest(*[f] * 2):
                    trigger = line1.split('\t')[-1].strip('\n')
                    code = line2.split('\t')[-1].strip('\n')
                    self.span_dict[code].append(trigger.lower())

    def construct_dicts(self):
        """
        Calls the methods which actually contruct the dictionaries for the data, labels, and label descriptions
        :return:
        """
        self.construct_data_dict(train=True)
        self.construct_data_dict()
        self.construct_data_dict(test=True)


        self.construct_label_dict(train=True)


        self.construct_label_dict()
        self.construct_span_dict(train=True)
        self.construct_span_dict()

        for k, v in self.span_dict.items():
            self.span_dict[k] = Counter(v)
        self.construct_label_desc_dict()

    def plot_label_dist(self, sorted_counts):
        # codes = [i+1 if i %20 == 0  else '' for i, x in enumerate(sorted_counts)]
        codes = [i + 1 for i in range(len(sorted_counts))]
        counts = [x[1] for x in sorted_counts]
        plt.plot(codes, counts)
        plt.xlabel("Frequency Ranks of Codes")
        plt.ylabel("# Docs Assigned a Given Code")
        plt.title("Code Frequencies")
        plt.show()

    def filter_labels(self):
        """
        Gets a code count across all data (train + dev) and discards any labels under a user-specified threshold
        :return: set: all labels which meet label threshold criteria
        """

        label_counts = Counter([item for sublist in self.label_dict.values() for item in sublist])
        if self.args.make_plots:
            sorted_lc = sorted(label_counts.items(), key=lambda pair: pair[1], reverse=True)
            self.plot_label_dist(sorted_lc)
        to_keep = {k for k, v in label_counts.items() if v > self.args.label_threshold}
        return to_keep

    def write_files(self):
        """
        Writes the .tsv files into train, dev, and test splits which are then loaded in again to create the pickled versions
        I like this step because I like being able to go into the data and actually see it
        :return:
        """
        self.construct_dicts()
        labels_to_keep = self.filter_labels()
        with open(out_path + cantemist_path + 'label_descriptions.tsv', 'w') as f:
            for code, desc in sorted(self.label_desc_dict.items()):
                if desc:
                    f.write(code + '\t' + desc + '\n')


        with open('processed_data/cantemist/test_gold_{}.tsv'.format(self.args.ignore_labelless_docs),
                  'w') as test_gold_out:
            test_gold_out.write('id\tlabels\n')

            for file in [self.train_file, self.dev_file, self.test_file]:
                test = False

                with open('processed_data/cantemist/{}.tsv'.format(file), 'w') as outf:
                    outf.write('id\ttext\tlabels\n')

                    if file == self.dev_file and self.args.train_on_all == False:
                        ids = self.dev_ids
                    elif file == self.train_file and self.args.train_on_all == False:
                        ids = self.train_ids
                    elif self.args.train_on_all == True and file == self.train_file:
                        ids = self.train_ids + self.dev_ids
                    elif self.args.train_on_all == True and file == self.dev_file:
                        ids = []
                    else:
                        ids = self.test_ids
                        test = True
                    for doc_id in ids:
                        text = self.data_dict[doc_id]
                        labels = '|'.join([l for l in self.label_dict[doc_id] if l in labels_to_keep])
                        if self.args.ignore_labelless_docs and not labels and not test:
                            self.n_disc_docs += 1
                            continue
                        else:
                            if test:
                                labels = '|'.join([l for l in self.label_dict[doc_id]])
                                test_gold_out.write(doc_id + '\t' + labels + '\n')
                                outf.write(doc_id + '\t' + text + '\n')
                            else:
                                outf.write(doc_id + '\t' + text + '\t' + labels + '\n')

    def process_data(self):
        """
        Read in processed data to binarize the labels
        :return:
        """
        if not os.path.exists('processed_data/cantemist/{}.tsv'.format(self.train_file)) or self.args.preprocess:
            self.write_files()
        else:
            self.construct_label_desc_dict()
        if self.args.train_on_all:
            to_iterate = [self.train_file, self.test_file]
        else:
            to_iterate = [self.train_file, self.dev_file, self.test_file]
        for data_type in to_iterate:
            file_path = 'processed_data/cantemist/{}.tsv'.format(data_type)
            data = pd.read_csv(file_path, sep='\t', skip_blank_lines=True)
            if data_type == self.test_file:
                labels = []
            else:
                labels = list(data['labels'])
                labels = [l.split('|') if type(l) == str else [] for l in labels]
            if data_type == self.train_file:
                labels_binarized = self.mlb.fit_transform(labels)
            else:
                labels_binarized = self.mlb.transform(labels)
            # elif self.args.train_on_all==True:
            #     labels_binarized = self.mlb.fit_transform(labels)

            labels_ranked = labels_binarized.copy()

            for l_str, l_bin in zip(labels, labels_ranked):
                for rank, l in enumerate(l_str):
                    idx = np.where(self.mlb.classes_ == l)
                    l_bin[idx] = len(l_str) - rank



            if data_type == self.test_file:
                data = [(data.iloc[idx, 0], data.iloc[idx, 1], None, None) for idx in range(len(data))]
            else:
                data = [(data.iloc[idx, 0], data.iloc[idx, 1], labels_binarized[idx, :], labels_ranked[idx,:]) for idx in range(len(data))]
            save('processed_data/cantemist/{}.p'.format(data_type), data)
            if not os.path.exists('processed_data/cantemist/mlb_{}_{}.p'.format(self.args.label_threshold,
                                                                                self.args.train_on_all)) and data_type == self.train_file:
                save('processed_data/cantemist/mlb_{}_{}.p'.format(self.args.label_threshold,
                                                                   self.args.train_on_all),
                     self.mlb)

                # get class weights
                total_train_docs = len(self.train_ids)
                self.class_weight_dict = {k: (total_train_docs - v) / v for k, v in self.class_weight_dict.items()}
                # and make sure they're in order according to the binarized labels....
                class_weights = [(k, v) for k, v in self.class_weight_dict.items() if k in set(self.mlb.classes_)]
                class_weights = sorted(class_weights, key=lambda x: list(self.mlb.classes_).index(x[0]))

                assert list(self.mlb.classes_) == [k for k, v in class_weights], print("Sorry, label order mismatch")
                # we only want the numbers now, not the codes themselves....
                class_weights = [v for c, v in class_weights]

                if not os.path.exists(
                        'processed_data/cantemist/class_weights_{}_{}.p'.format(str(len(self.class_weight_dict)),
                                                                                self.train_on_all)):
                    save('processed_data/cantemist/class_weights_{}_{}.p'.format(str(len(self.mlb.classes_)),
                                                                                 self.train_on_all),
                         class_weights)


                # get the label descriptions to save and make sure they're in the same order as the binarized labels
                label_descs_to_save = [(k, v) for k, v in self.label_desc_dict.items() if k in set(self.mlb.classes_)]

                label_descs_to_save = sorted(label_descs_to_save, key=lambda x: list(self.mlb.classes_).index(x[0]))

                assert list(self.mlb.classes_) == [k for k, v in label_descs_to_save], print("Sorry, label order mismatch")

                if not os.path.exists('processed_data/cantemist/label_desc_{}.p'.format(self.args.label_threshold)):
                    save('processed_data/cantemist/label_desc_{}.p'.format(self.args.label_threshold),
                         label_descs_to_save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_threshold",
        default=0,
        type=int,
        help="Exclude labels which occur <= threshold",
    )
    parser.add_argument('--ignore_labelless_docs', action='store_true',
                        help="Whether to ignore documents with no labels.")
    parser.add_argument('--preprocess', action='store_true', help="Whether to redo all of the pre-processing.")
    parser.add_argument('--make_plots', action='store_true', help="Whether to make plots on data.")
    parser.add_argument('--train_on_all', action='store_true')
    parser.add_argument('--data_dir', default='processed_data/cantemist/')
    parser.add_argument('--local_rank', default=-1)
    parser.add_argument('--model_name_or_path', type=str, default='xlm-roberta-base')
    parser.add_argument('--doc_max_seq_length', default=256)

    parser.add_argument(
        "--model_type",
        default='xlmroberta',
        type=str,
    )


    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--prediction_threshold",
        default=0.5,
        type=float,
        help="Threshold at which to decide between 0 and 1 for labels.",
    )
    parser.add_argument(
        "--loss_fct", default="none", type=str, help="The function to use.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--label_max_seq_length",
        default=15,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--logit_aggregation", type=str, default='max', help="Whether to aggregate logits by max value "
                                                                             "or average value. Options:"
                                                                             "'--max', '--avg'")
    parser.add_argument("--label_attention", action="store_true", help="Whether to use the label attention model")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument('--do_iterative_class_weights', action='store_true', help="Whether to use iteratively "
                                                                                  "calculated class weights")
    parser.add_argument('--do_normal_class_weights', action='store_true', help="Whether to use normally "
                                                                               "calculated class weights")
    parser.add_argument('--do_ranking_loss', action='store_true', help="Whether to use the ranking loss component.")
    parser.add_argument('--do_weighted_ranking_loss', action='store_true',
                        help="Whether to use the weighted ranking loss component.")
    parser.add_argument('--do_experimental_ranks_instead_of_labels', action='store_true', help='Whether to send ranks '
                                                                                               'instead of binary labels to loss function')
    parser.add_argument('--doc_batching', action='store_true', help="Whether to fit one document into a batch during")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup proportion.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=21, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")




    args = parser.parse_args()
    cantemist_processor = CantemistReader(args)
    cantemist_processor.process_data()

    try:
        processor = MyProcessor(args)
    except:
        cantemist_reader = CantemistReader(args)
        cantemist_reader.process_data()
        # spanish_reader = SpanishReader(args)
        # spanish_reader.process_data()
        # german_reader = GermanReader(args)
        # german_reader.process_data()
        processor = MyProcessor(args)
    class_weights = processor.get_class_weights()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',
        do_lower_case=False,
        cache_dir=None,
    )


    train_dataset, label_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=False, label_data=True)