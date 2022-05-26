import json
import os
from functools import partial

import numpy as np


def _evaluate_count_empty(pred_labels, gold_labels):
    if len(pred_labels) == 0:
        pred_labels.add('empty')
    if len(gold_labels) == 0:
        gold_labels.add('empty')
    tp = len(pred_labels & gold_labels)
    r = tp / len(gold_labels)
    p = tp / len(pred_labels)
    try:
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1 = 0
    return p, r, f1


def _evaluate_notcount_empty(pred_labels, gold_labels):
    tp = len(pred_labels & gold_labels)
    try:
        r = tp / len(gold_labels)
        p = tp / len(pred_labels)
        f1 = 2 * p * r / (p + r)
    except ZeroDivisionError:
        p = 0
        r = 0
        f1 = 0
    return p, r, f1


def _evaluate(pred_labels, gold_labels, count_empty):
    if count_empty:
        return _evaluate_count_empty(pred_labels, gold_labels)
    else:
        return _evaluate_notcount_empty(pred_labels, gold_labels)


def _construct_prefixs(labels):
    prefixs = dict()
    for label in labels:
        prefix, status = label.split('-')
        status = status.split(':')[-1]
        try:
            prefixs[prefix].add(status)
        except KeyError:
            prefixs[prefix] = {status}
    return prefixs


def _merge(previous_statuses_w, current_statuses_w):
    if '阳性' in previous_statuses_w and '阴性' in current_statuses_w:
        previous_statuses_w.remove('阳性')
    if '阴性' in previous_statuses_w and '阳性' in current_statuses_w:
        previous_statuses_w.remove('阴性')
    if '医生阳性' in previous_statuses_w and '医生阴性' in current_statuses_w:
        previous_statuses_w.remove('医生阳性')
    if '医生阴性' in previous_statuses_w and '医生阳性' in current_statuses_w:
        previous_statuses_w.remove('医生阴性')
    if '未知' in previous_statuses_w and len(current_statuses_w) > 0:
        previous_statuses_w.remove('未知')
    if len(previous_statuses_w) > 0 and '未知' in current_statuses_w:
        current_statuses_w.remove('未知')
    merged_statuses_w = previous_statuses_w | current_statuses_w
    return merged_statuses_w


def merge(previous_labels_w, current_labels_w):
    previous_prefixs = _construct_prefixs(previous_labels_w)
    current_prefixs = _construct_prefixs(current_labels_w)
    for key in current_prefixs.keys():
        if key not in previous_prefixs:
            previous_prefixs[key] = current_prefixs[key]
        else:
            previous_prefixs[key] = _merge(previous_prefixs[key], current_prefixs[key])
    merged_labels_w = set()
    for key in previous_prefixs.keys():
        for status in previous_prefixs[key]:
            merged_labels_w.add('{}-{}'.format(key, "状态:"+status))
    return merged_labels_w


def preprocess(model, name, batch_size):
    ontology = model.ontology
    status_num = len(ontology.ontology_dict[ontology.mutual_slot])

    slots_pred_labels_v, slots_gold_labels_v = model.inference(name, -1, batch_size)

    pred_labels_v = np.concatenate(slots_pred_labels_v, -1)  
    gold_labels_v = np.concatenate(slots_gold_labels_v, -1)

    size = pred_labels_v.shape[0]
    pred_labels_w = []
    gold_labels_w = []
    for i in range(size):
        pred_label_w = ontology.vec2label(pred_labels_v[i])
        gold_label_w = ontology.vec2label(gold_labels_v[i])
        pred_labels_w.append(pred_label_w)
        gold_labels_w.append(gold_label_w)
    return pred_labels_w, gold_labels_w
def preprocessx(model, name, batch_size):
    ontology = model.ontology
    status_num = len(ontology.ontology_dict[ontology.mutual_slot])

    score,slots_pred_labels_v, slots_gold_labels_v = model.inference_with_score(name, -1, batch_size)

    pred_labels_v = np.concatenate(slots_pred_labels_v, -1) 
    gold_labels_v = np.concatenate(slots_gold_labels_v, -1)
    score = np.concatenate(score, -1)
    size = pred_labels_v.shape[0]
    pred_labels_w = []
    gold_labels_w = []

    for i in range(size):

        gold_label_w = ontology.vec2label(gold_labels_v[i])
        pred_labels_w.append(ontology.vec2labelx(pred_labels_v[i],score[i]))
        gold_labels_w.append(gold_label_w)
    
    return pred_labels_w, gold_labels_w

def _window_eval(window_pred_labels_w, window_gold_labels_w, count_empty, func,filename=None):
    ps = []
    rs = []
    f1s = []
    for pred_label_w, gold_label_w in zip(window_pred_labels_w, window_gold_labels_w):
        pred_label_w = set(map(func, pred_label_w))
        gold_label_w = set(map(func, gold_label_w))
        p, r, f1 = _evaluate(pred_label_w, gold_label_w, count_empty)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)
    infos = {
        'p': p,
        'r': r,
        'f1': f1
    }
    return infos

def save_file(pack,filename,Data):
    with open(os.path.join(os.path.join('./data/',pack), filename), "w", encoding='utf-8') as f:
        f.write(json.dumps(Data, ensure_ascii=False, indent=5))
def _dialog_eval(window_pred_labels_w, window_gold_labels_w, dialogs, count_empty, func,filename=None):
    i = 0
    ps = []
    rs = []
    f1s = []
    Data = list()
    Window_level_Data = list()
    for dialog in dialogs:
        dialog_pred_labels_w = set()
        dialog_gold_labels_w = set()
        windows = list()

        Dia = list()
        dic = dict()
        for window in dialog:
            win = dict()
            Dia.append(window['utterances'][-1])
            win['window'] = window
            win['pred_labels_w'] = list(set(map(func, window_pred_labels_w[i])))
            win['gold_labels_w'] = list(set(map(func, window_gold_labels_w[i])))
            windows.append(win)
            dialog_pred_labels_w = merge(dialog_pred_labels_w, set(window_pred_labels_w[i]))
            dialog_gold_labels_w = merge(dialog_gold_labels_w, set(window_gold_labels_w[i]))
            i += 1
        Window_level_Data.append(windows)
        dialog_pred_labels_w = set(map(func, dialog_pred_labels_w))
        dialog_gold_labels_w = set(map(func, dialog_gold_labels_w))
        dic['Dialog'] = Dia
        dic['dialog_pred_labels_w'] = list(dialog_pred_labels_w)
        dic['dialog_gold_labels_w'] = list(dialog_gold_labels_w)
        dic['wrong_labels'] = list(dialog_pred_labels_w-(dialog_pred_labels_w&dialog_gold_labels_w))
        Data.append(dic)
        p, r, f1 = _evaluate(dialog_pred_labels_w, dialog_gold_labels_w, count_empty)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    save_file("Dialog", filename, Data)
    save_file("Window",filename,Window_level_Data)
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)

    infos = {
        'p': p,
        'r': r,
        'f1': f1
    }
    return infos


_get_category = lambda x: x.split('-')[0].split(':')[0]
_get_item = lambda x: x.split('-')[0]
_get_full = lambda x: x

window_category = partial(_window_eval, func=_get_category,filename="category.json")
window_item = partial(_window_eval, func=_get_item,filename="category_item.json")
window_full = partial(_window_eval, func=_get_full,filename="full.json")

def evaluate(model, name, batch_size, count_empty=True):
    dialogs = model.data.datasets[name]['origin']
    window_pred_labels_w, window_gold_labels_w = preprocess(model, name, batch_size)
    infos = {}
    infos['category'] = window_category(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['item'] = window_item(window_pred_labels_w, window_gold_labels_w, count_empty)
    infos['full'] = window_full(window_pred_labels_w, window_gold_labels_w, count_empty)

    return infos
if __name__ == '__main__':
    pass
