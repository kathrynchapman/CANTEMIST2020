from collections import defaultdict
from torch import save, load
from functools import partial
import pandas as pd
from beautifultable import BeautifulTable
from datetime import datetime

models = ['baseline', 'label_attention']
num_epohcs = ['any int']
doc_batching = ['no_doc_batching', 'doc_batching_max']
ranking_loss = ['no_ranking_loss', 'ranking_loss', 'weighted_ranking_loss']
class_weights = ['no_class_weights', 'dynamic_class_weights', 'static_class_weights']
loss_function = ['bce', 'bbce', 'none', 'bce ranks', 'bbce ranks']


def get_input(ref_type, str_type):
    """
    Gets all of the deets to for an entry (to add, to delete, etc)
    :param ref_type: list; all of the options for that parameter
    :param str_type: str; basically the name of the ref type
    :return: inp_type: str; the data
    """
    invalid = True
    while invalid:
        inp = input("\nEnter {}, [{}]: ".format(str_type, ', '.join(ref_type)))
        if inp == 'exit':
            return inp
        try:
            inp_type = ref_type[int(inp) - 1]
            print("You entered ", inp_type)
        except:
            inp_type = inp
        if inp_type not in ref_type and str_type != 'number epochs':
            print("Sorry, that isn't a valid input.")
        else:
            invalid = False
    return inp_type


def fill_dict(exp_dict):
    """
    Builds the nested dictionary by adding new data
    :param exp_dict: the dict (duh)
    :return: None
    """
    blah = 'a'
    done = False
    while not done:
        add_to_notes = ''
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        ref_type = [models, num_epohcs, doc_batching, ranking_loss, class_weights, loss_function]
        str_type = ['model type', 'number epochs', 'doc batching', 'ranking loss', 'class weights', 'loss function']
        inputs = []
        for r_type, s_type in zip(ref_type, str_type):
            inp = get_input(r_type, s_type)
            if inp == 'exit':
                blah = True
                break
            else:
                inputs.append(inp)
        if blah == True:
            break

        if inputs[4] != 'no_class_weights' and inputs[5] == 'bbce':
            reduct = input("How did you combine bbce and class weights loss? [mean, sum]: ")
            reduct = 'mean' if reduct == 1 else reduct
            reduct = 'sum' if reduct == 2 else reduct
            inputs[5] += ' - ' + reduct

        if exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['MAP']:
            valid = False
            while not valid:
                cont = input(
                    "You already have an entry for this - do you want to continue and overwrite? [y, n, just notes]: ")
                if cont not in ['y', 'n', 'just notes'.strip(), '1', '2', '3']:
                    print("Sorry, didn't get that...")
                else:
                    valid = True
            if cont == 'n' or cont == '2':
                continue
            elif cont == 'y' or cont == '1':
                add_to_notes = " (last edited {})".format(dt_string)
            elif cont == 'just notes' or cont == '3':
                notes = input("Just type the notes here: ")
                if notes == 'exit':
                    break
                exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['Notes'] = notes + \
                                                                                                      " (last edited {})".format(
                                                                                                          dt_string)
                save(exp_dict, 'exp_dict.p')
                continue

        MAP = input("Enter MAP: ")
        F1 = input("Enter F1: ")
        P = input("Enter P: ")
        R = input("Enter R: ")
        notes = input("Any notes on this? (Just type them if so): ")

        exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['MAP'] = float(MAP)
        exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['F1'] = float(F1)
        exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['P'] = float(P)
        exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['R'] = float(R)
        exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['Entry Date'] = dt_string

        if notes or add_to_notes:
            exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]]['Notes'] = notes + add_to_notes

        save(exp_dict, 'exp_dict.p')
        done = True if input("Do you have another experiment to enter? [y, n]: ") == 'n' else False
        if not done:
            print('-' * 100)


def lookup(exp_dict):
    """
    Looks up an experiment in the dict and prints the results
    :param exp_dict:
    :return:
    """
    done = False
    while not done:
        model_type = get_input(models, 'model type')
        n_eps = get_input(num_epohcs, 'number epochs')
        doc_batches = get_input(doc_batching, 'doc batching')
        rank_loss = get_input(ranking_loss, 'ranking loss')
        cls_wts = get_input(class_weights, 'class weights')
        loss_fct = get_input(loss_function, 'loss function')

        print("Results for {}, {}, {}, {}, {}, {}".format(model_type, n_eps, doc_batches, rank_loss, cls_wts, loss_fct))
        print("MAP:", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['MAP'])
        print("F1:", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['F1'])
        print("P:", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['P'])
        print("R:", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['R'])
        print("Notes: ", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['Notes'])
        print("Entry Date: ", exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['Entry Date'])
        done = True if input("Do you have another experiment to look up? [y, n]: ") == 'n' else False
        if not done:
            print('-' * 100)


def viewall(exp_dict):
    """
    Prints a *beautiful* table of all of the entered data
    :param exp_dict:
    :return:
    """
    '''
    # >>> from beautifultable import BeautifulTable
    # >>> table = BeautifulTable()
    # >>> table.rows.append(["Jacob", 1, "boy"])
    # >>> table.rows.append(["Isabella", 1, "girl"])
    # >>> table.rows.append(["Ethan", 2, "boy"])
    # >>> table.rows.append(["Sophia", 2, "girl"])
    # >>> table.rows.append(["Michael", 3, "boy"])
    # >>> table.rows.header = ["S1", "S2", "S3", "S4", "S5"]
    # >>> table.columns.header = ["name", "rank", "gender"]
    # >>> print(table)
    :param exp_dict:
    :return:
    '''
    table = BeautifulTable(maxwidth=300)
    table.columns.header = ["Model", "#Epochs", "Doc Batching", "Ranking Loss", "Class Weights", "Loss Funct", "MAP",
                            "F1", "P", "R", "Notes", "Entry Date"]
    table.column_widths = [15, 9, 15, 15, 23, 15, 7, 7, 7, 7, 20, 15]
    for m in exp_dict.keys():
        for n_ep in exp_dict[m].keys():
            for db in exp_dict[m][n_ep].keys():
                for rl in exp_dict[m][n_ep][db].keys():
                    for cw in exp_dict[m][n_ep][db][rl].keys():
                        for lf in exp_dict[m][n_ep][db][rl][cw].keys():
                            map = exp_dict[m][n_ep][db][rl][cw][lf]['MAP']
                            f1 = exp_dict[m][n_ep][db][rl][cw][lf]['F1']
                            p = exp_dict[m][n_ep][db][rl][cw][lf]['P']
                            r = exp_dict[m][n_ep][db][rl][cw][lf]['R']
                            notes = exp_dict[m][n_ep][db][rl][cw][lf]['Notes']
                            ed = exp_dict[m][n_ep][db][rl][cw][lf]['Entry Date']

                            m_ = 'label attn' if m == 'label_attention' else m
                            rl_ = 'Y' if rl == 'ranking_loss' else 'N'
                            db_ = 'N' if db == 'no_doc_batching' else db[-3:]
                            cw_ = 'N' if cw == 'no_class_weights' else cw
                            ed = 'N/A' if ed == 0.0 else ed

                            table.rows.append([m_, n_ep, db_, rl_, cw_, lf, map, f1, p, r, notes, ed])
    print(table)
    # print("Results for {}, {}, {}, {}, {}, {}".format(m, n_ep, db, rl, cw, lf))
    # print("MAP:", exp_dict[m][n_ep][db][rl][cw][lf]['MAP'])
    # print("F1:", exp_dict[m][n_ep][db][rl][cw][lf]['F1'])
    # print("P:", exp_dict[m][n_ep][db][rl][cw][lf]['P'])
    # print("R:", exp_dict[m][n_ep][db][rl][cw][lf]['R'])


def delete(exp_dict, backup_dict):
    """
    Deletes a dictionary entry
    :param exp_dict:
    :param backup_dict:
    :return:
    """
    """
    implement a delete function
    :param exp_dict:
    :return:
    """
    # viewall(exp_dict)
    # exp_dict['label_attention']['45']['doc_batching_max']['ranking_loss'].pop('class_weights')
    # viewall(exp_dict)
    # save(exp_dict, 'exp_dict.p')
    blah = 'a'
    done = False
    while not done:
        add_to_notes = ''
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        ref_type = [models, num_epohcs, doc_batching, ranking_loss, class_weights, loss_function]
        str_type = ['model type', 'number epochs', 'doc batching', 'ranking loss', 'class weights', 'loss function']
        inputs = []
        for r_type, s_type in zip(ref_type, str_type):
            inp = get_input(r_type, s_type)
            if inp == 'exit':
                blah = True
                break
            else:
                inputs.append(inp)
        if blah == True:
            break

        to_delete = exp_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]].pop(inputs[5])
        backup_dict[inputs[0]][inputs[1]][inputs[2]][inputs[3]][inputs[4]][inputs[5]] = to_delete
        save(exp_dict, 'exp_dict.p')
        save(backup_dict, 'backup.p')
        done = input("Do you have another entry to delete? [y, n] ")
        done = True if done == 'y' else False

def show_remaining_exps(exp_dict):
    """
    Basically just print out a table of all of the combinations of parameters not currently in the dictionary;
    useful for brainstorming what to do next when you're utterly, utterly lost about how to proceed
    :param exp_dict:
    :return:
    """
    '''
        # >>> from beautifultable import BeautifulTable
        # >>> table = BeautifulTable()
        # >>> table.rows.append(["Jacob", 1, "boy"])
        # >>> table.rows.append(["Isabella", 1, "girl"])
        # >>> table.rows.append(["Ethan", 2, "boy"])
        # >>> table.rows.append(["Sophia", 2, "girl"])
        # >>> table.rows.append(["Michael", 3, "boy"])
        # >>> table.rows.header = ["S1", "S2", "S3", "S4", "S5"]
        # >>> table.columns.header = ["name", "rank", "gender"]
        # >>> print(table)
        :param exp_dict:
        :return:
    '''
    table = BeautifulTable(maxwidth=300)
    table.columns.header = ["Model", "#Epochs", "Doc Batching", "Ranking Loss", "Class Weights", "Loss Funct"]
    table.column_widths = [17, 9, 20, 23, 23, 15]
    print("######################################## Experiments to Run ########################################")
    for m in models:
        for n_ep in num_epohcs:
            for db in doc_batching:
                for rl in ranking_loss:
                    for cw in class_weights:
                        for lf in loss_function:
                            if not exp_dict[m]['45'][db][rl][cw][lf]['MAP']:
                                if not (lf == 'none' and rl == 'no_ranking_loss'):
                                    if not (rl == 'weighted_ranking_loss' and cw == 'no_class_weights'):
                                        if not (rl!= 'weighted_ranking_loss' and lf[-5:] == 'ranks'):
                                            table.rows.append([m, n_ep, db, rl, cw, lf])
    print(table)


if __name__ == '__main__':
    try:
        exp_dict = load('exp_dict.p')
    except:
        exp_dict = defaultdict(
            partial(defaultdict,
            partial(defaultdict,
            partial(defaultdict,
            partial(defaultdict,
            partial(defaultdict,
            partial(defaultdict, float)))))))

    finished = False
    while not finished:
        to_do = input(
            "\t(1)\t Enter new data, "
            "\n\t(2)\t look up existing data, "
            "\n\t(3)\t view all existing data, "
            "\n\t(4)\t delete an entry,"
            "\n\t(5)\t or view possible experiments? "
            "\n\t (hit any other key to quit)\n")
        if to_do == '1':
            fill_dict(exp_dict)
        elif to_do == '2':
            lookup(exp_dict)
        elif to_do == '3':
            viewall(exp_dict)
        elif to_do == '4':
            try:
                backup_dict = load('backup.p')
            except:
                backup_dict = defaultdict(
                    partial(defaultdict,
                    partial(defaultdict,
                    partial(defaultdict,
                    partial(defaultdict,
                    partial(defaultdict,
                    partial(defaultdict, float)))))))
            delete(exp_dict, backup_dict)
        elif to_do == '5':
            show_remaining_exps(exp_dict)
        else:
            finished = True


