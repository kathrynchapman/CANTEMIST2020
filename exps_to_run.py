from collections import defaultdict
from torch import save, load
from functools import partial

models = ['baseline', 'label_attention']

num_epohcs = ['25', '45']

doc_batching = ['doc_batching_max', 'doc_batching_mean', 'no_doc_batching']

ranking_loss = ['ranking_loss', 'no_ranking_loss']

class_weights = ['dynamic_class_weights', 'static_class_weights', 'no_class_weights']

loss_function = ['bce', 'bbce']

try:
    exp_dict = load('exp_dict.p')
except:
    # exp_dict = defaultdict(partial(defaultdict(partial(
    #     defaultdict(partial(defaultdict(partial(defaultdict(partial(defaultdict(partial(defaultdict, float))))))))))))

    exp_dict = defaultdict(
                partial(defaultdict,
                partial(defaultdict,
                partial(defaultdict,
                partial(defaultdict,
                partial(defaultdict,
                partial(defaultdict, float)))))))

    # d = defaultdict(partial(defaultdict, int))
    # exp_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:
    #     defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))))


# print("######################################## Experiments to Run ########################################")
#
# for m in models:
#     for n_ep in num_epohcs:
#         for db in doc_batching:
#             for rl in ranking_loss:
#                 for cw in class_weights:
#                     for lf in loss_function:
#                         # exp_dict[m][n_ep][db][rl][cw][lf]['MAP'] = ''
#                         # exp_dict[m][n_ep][db][rl][cw][lf]['F1'] = ''
#                         # exp_dict[m][n_ep][db][rl][cw][lf]['P'] = ''
#                         # exp_dict[m][n_ep][db][rl][cw][lf]['R'] = ''

# {'MAP': '', 'F1': '', 'P': '', 'R': ''}
# print(m + '\t' + n_ep + '\t' + db + '\t' + rl + '\t' + cw + '\t' + lf)

# print(exp_dict)

def get_input(ref_type, str_type):
    invalid = True
    while invalid:
        inp = input("Enter {}, [{}]: ".format(str_type, ', '.join(ref_type)))
        try:
            inp_type = ref_type[int(inp) - 1]
            print("You entered ", inp_type)
        except:
            inp_type = inp
        if inp_type not in ref_type:
            print("Sorry, that isn't a valid input.")
        else:
            invalid = False
    return inp_type


done = False
while not done:
    model_type = get_input(models, 'model type')
    n_eps = get_input(num_epohcs, 'number epochs')
    doc_batches = get_input(doc_batching, 'doc batching')
    rank_loss = get_input(ranking_loss, 'ranking loss')
    cls_wts = get_input(class_weights, 'class weights')
    loss_fct = get_input(loss_function, 'loss function')

    if exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['MAP']:
        cont = input("You already have an entry for this - do you want to continue and overwrite? [y, n]: ")
        if cont == 'n':
            continue

    MAP = input("Enter MAP: ")
    F1 = input("Enter F1: ")
    P = input("Enter P: ")
    R = input("Enter R: ")
    notes = input("Any notes on this? (Just type them if so): ")



    exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['MAP'] = float(MAP)
    exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['F1'] = float(F1)
    exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['P'] = float(P)
    exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['R'] = float(R)
    if notes:
        exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['Notes'] = notes

    save(exp_dict, 'exp_dict.p')
    done = True if input("Do you have another experiment to enter? [y, n]: ") == 'n' else False
    if not done:
        print('-'*100)

# ################### Begin Interactive Dict Construction ###################
# done = False
# invalid = True
# while not done:
#     invalid = True
#     while invalid:
#         inp = input("Enter model type, [{}]: ".format(', '.join(models)))
#         try:
#             model_type = models[int(inp)-1]
#         except:
#             model_type = inp
#         if model_type not in models:
#             print("Sorry, that isn't a valid input.")
#         else:
#             invalid = False
#
#
#
#
#
#
#     model_type = models[int(input("Enter model type, [{}]: ".format(', '.join(models)))) - 1]
#     print("You entered ", model_type)
#     n_eps = num_epohcs[int(input("Enter number epochs, [{}]: ".format(', '.join(num_epohcs)))) - 1]
#     print("You entered ", n_eps)
#     doc_batches = doc_batching[int(input("Enter doc batching, [{}]: ".format(', '.join(doc_batching)))) - 1]
#     print("You entered ", doc_batches)
#     rank_loss = ranking_loss[int(input("Enter ranking loss, [{}]: ".format(', '.join(ranking_loss))))-1]
#     print("You entered ", rank_loss)
#     cls_wts = class_weights[int(input("Enter class weights, [{}]: ".format(', '.join(class_weights))))-1]
#     print("You entered ", cls_wts)
#     loss_fct = loss_function[int(input("Enter loss function, [{}]: ".format(', '.join(loss_function))))-1]
#     print("You entered ", loss_fct)
#     MAP = input("Enter MAP: ")
#     F1 = input("Enter F1: ")
#     P = input("Enter P: ")
#     R = input("Enter R: ")
#     notes = input("Any notes on this? (Just type them if so): ")
#
#     exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['MAP'] = float(MAP)
#     exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['F1'] = float(F1)
#     exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['P'] = float(P)
#     exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['R'] = float(R)
#     if notes:
#         exp_dict[model_type][n_eps][doc_batches][rank_loss][cls_wts][loss_fct]['Notes'] = notes
#
#     done = True if input("Do you have another experiment to enter? [y, n]: ") == 'n' else False


print("######################################## Experiments to Run ########################################")

for m in models:
    for n_ep in num_epohcs:
        for db in doc_batching:
            for rl in ranking_loss:
                for cw in class_weights:
                    for lf in loss_function:
                        if not exp_dict[m][n_ep][db][rl][cw][lf]['MAP']:
                            print(m + '\t' + n_ep + '\t' + db + '\t' + rl + '\t' + cw + '\t' + lf)
                        # exp_dict[m][n_ep][db][rl][cw][lf]['F1'] = ''
                        # exp_dict[m][n_ep][db][rl][cw][lf]['P'] = ''
                        # exp_dict[m][n_ep][db][rl][cw][lf]['R'] = ''


