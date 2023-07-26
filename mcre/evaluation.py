import json

import numpy as np


def get_f1(key, prediction):
    correct_by_relation = ((key == prediction) & (prediction != 0)).astype(np.int32).sum()
    guessed_by_relation = (prediction != 0).astype(np.int32).sum()
    gold_by_relation = (key != 0).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro, correct_by_relation, guessed_by_relation, gold_by_relation


def get_indiv_f1(num_class, key, prediction):
    ret = {}
    for i in range(1, num_class):
        correct_by_relation = ((key == i) & (key == prediction)).astype(np.int32).sum()
        guessed_by_relation = (prediction == i).astype(np.int32).sum()
        gold_by_relation = (key == i).astype(np.int32).sum()

        prec_micro = 1.0
        if guessed_by_relation > 0:
            prec_micro = float(correct_by_relation) / float(guessed_by_relation)
        recall_micro = 1.0
        if gold_by_relation > 0:
            recall_micro = float(correct_by_relation) / float(gold_by_relation)
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

        ret[i] = {'precision': round(prec_micro * 100, 2), 'recall': round(recall_micro * 100, 2), 'f1': round(f1_micro * 100, 2),
                  'correct': correct_by_relation, 'guessed': guessed_by_relation, 'gold': gold_by_relation}

    return ret


def generate_predicted_file(input_file, output_file, rel2id, key, prediction):
    with open(input_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    id2rel = {v: k for k, v in rel2id.items()}
    new_data = []
    for d, k, p in zip(data, list(key), list(prediction)):
        if 'revised_relation' in d.keys():
            assert rel2id[d['revised_relation']] == k
        else:
            assert rel2id[d['relation']] == k

        mc_pred_relation = id2rel[p]
        d['mc_pred_relation'] = mc_pred_relation
        new_data.append(d)

    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(new_data, fp)

# def get_accuracy(key_, key, pred):
#     if len(key_) == 0:
#         return 0, 0
#     num_fp = ((key_ != 0) & (key != key_)).astype(np.int32).sum()
#     correct_fp = ((key_ != 0) & (key != key_) & (pred == key)).astype(np.int32).sum()
#     fp_accuracy = float(correct_fp) / float(num_fp)
#     num_tp = ((key_ != 0) & (key == key_)).astype(np.int32).sum()
#     correct_tp = ((key_ != 0) & (key == key_) & (pred == key)).astype(np.int32).sum()
#     tp_accuracy = float(correct_tp) / float(num_tp)
#     return fp_accuracy, tp_accuracy
