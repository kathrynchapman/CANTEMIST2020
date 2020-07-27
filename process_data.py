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
        self.data_dict = dict()
        self.label_dict = defaultdict(list)
        self.train_ids = []
        self.dev_ids = []
        self.test_ids = []
        self.mlb = MultiLabelBinarizer()
        self.args = args
        self.label_desc_dict = defaultdict(str)
        self.span_dict = defaultdict(list)

        self.train_file = 'train_{}_{}'.format(self.args.label_threshold, self.args.ignore_labelless_docs)
        self.dev_file = 'dev_{}_{}'.format(self.args.label_threshold, self.args.ignore_labelless_docs)
        self.test_file = 'test_{}_{}'.format(self.args.label_threshold, self.args.ignore_labelless_docs)
        self.n_disc_docs = 0
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
                    self.label_dict[doc_id].append(label)
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

                    if file == self.dev_file:
                        ids = self.dev_ids
                    elif file == self.train_file:
                        ids = self.train_ids
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
        for data_type in [self.train_file, self.dev_file, self.test_file]:
        # for data_type in [self.train_file]:
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




            # print(min(sum(l) for l in labels_binarized))
            # print(max(sum(l) for l in labels_binarized))

            labels_ranked = labels_binarized.copy()

            for l_str, l_bin in zip(labels, labels_ranked):
                idx2rank = {}
                # print(l_str)
                # print(l_bin.tolist())
                for rank, l in enumerate(l_str):
                    idx = np.where(self.mlb.classes_ == l)
                    l_bin[idx] = len(l_str) - rank
                # print(l_str)
                # print(l_bin.tolist())

                # print(l_bin.tolist()[0])
            # print(labels_binarized)
            # labels_binarized = np.array([normalize(l_bin.reshape(1, -1), norm="max") for l_bin in labels_binarized])
            # # l_bin = normalize(l_bin.reshape(1, -1), norm="max")
            # for l_bin in labels_binarized:
            #     l_bin[l_bin == 0] = -1
            # for one, two in zip(labels_binarized, labels_binary):
            #     print(one.tolist())
            #     print(two.tolist())
            #     print('_'*100)






            label_descs_to_save = [(k, v) for k, v in self.label_desc_dict.items() if k in set(self.mlb.classes_)]

            label_descs_to_save = sorted(label_descs_to_save, key=lambda x: list(self.mlb.classes_).index(x[0]))

            assert list(self.mlb.classes_) == [k for k, v in label_descs_to_save], print("Sorry, label order mismatch")

            if data_type == self.test_file:
                data = [(data.iloc[idx, 0], data.iloc[idx, 1], None, None) for idx in range(len(data))]
            else:
                data = [(data.iloc[idx, 0], data.iloc[idx, 1], labels_binarized[idx, :], labels_ranked[idx,:]) for idx in range(len(data))]
            # print(data)

            save('processed_data/cantemist/{}.p'.format(data_type), data)
            if not os.path.exists('processed_data/cantemist/mlb_{}_{}.p'.format(self.args.label_threshold,
                                                                                self.args.ignore_labelless_docs)):
                save('processed_data/cantemist/mlb_{}_{}.p'.format(self.args.label_threshold,
                                                                   self.args.ignore_labelless_docs),
                     self.mlb)
            if not os.path.exists('processed_data/cantemist/label_desc_{}.p'.format(self.args.label_threshold,
                                                                                    self.args.ignore_labelless_docs)):
                save('processed_data/cantemist/label_desc_{}.p'.format(self.args.label_threshold,
                                                                       self.args.ignore_labelless_docs),
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
    args = parser.parse_args()
    cantemist_processor = CantemistReader(args)
    cantemist_processor.process_data()

    # cantemist_processor.construct_dicts()
    # cantemist_processor.filter_labels()
    # temp = cantemist_processor.span_dict.copy()
    # for code, spans in cantemist_processor.span_dict.items():
    #     if code[-1] == "H":
    #         print("Code:", code)
    #         print("Spans:", spans)
    #         print("Code:", code[:-2])
    #         print("Spans:", temp[code[:-2]])
    #         print("Official desc:", cantemist_processor.label_desc_dict[code[:-2]])
    #         print("-"*50)

    # label_counts = cantemist_processor.filter_labels()
    # label_counts = sorted(label_counts.items(), key=lambda pair: pair[1], reverse=True)
    # inverted_label_counts = defaultdict(int)
    # for label, count in label_counts.items():
    #     inverted_label_counts[count] += 1
    # print(label_counts[:100])
    # print(sorted(inverted_label_counts.items(), key=lambda pair: pair[0]))

    # avg_n_labels = np.average([len(labels) for labels in cantemist_processor.label_dict.values()])
    # print("Average n labels:", avg_n_labels)
