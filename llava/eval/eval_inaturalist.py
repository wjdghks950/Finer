import sys
import re
import os
import numpy as np
import json
import string
from collections import Counter
import pickle

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # if normalized_ground_truth in normalized_prediction:
    #     return 1.0, 1.0, 1.0

    ZERO_METRIC = (0, 0, 0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1.0
    else:
        return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval(pred, gold):
    f1, precision, recall = f1_score(pred, gold)
    em = exact_match_score(pred, gold)
    return f1, em


def remove_ans_prefix(text):
    ans_start_idx = text.find("Answer:")
    start_idx = 0 if ans_start_idx == -1 else ans_start_idx + len("Answer:")
    ans_text = text[start_idx:].strip().replace("</s>", "")
    return ans_text
    

if __name__ == '__main__':
    task_type = "high_coarse"
    data_dir = "/home/jk100/data/data/inaturalist"
    data_path = "val_noplant_64.json"

    preds_dir = "../preds"
    out_path = f"inaturalist_{task_type}_output.json"

    # Load dataset from iNaturalist
    if os.path.exists(os.path.join(data_dir, data_path)):
        with open(os.path.join(data_dir, data_path), "r") as fp:
            dataset = json.load(fp)
        fp.close()

    # Load predicted outputs from `preds`
    if os.path.exists(os.path.join(preds_dir, out_path)):
        with open(os.path.join(preds_dir, out_path), "r") as fp:
            outputs = json.load(fp)
        fp.close()

    f1_scores = []
    em_scores = []
    for idx, pred_text in outputs.items():
        idx = int(idx)
        annotation = dataset['annotations'][idx]
        img_id = annotation['image_id']
        cidx = annotation['category_id']  # category index
        category_dict = dataset['categories'][cidx]
        if task_type == "high_coarse":
            lbl = category_dict['supercategory'].lower().strip()
        elif task_type == "coarse":
            lbl = category_dict['common_name'].lower().strip()
        elif task_type == "fine":
            lbl = category_dict['name'].lower().strip()
        # TODO: Load the dataset and evaluate the output
        pred_text = remove_ans_prefix(pred_text)
        print(f"{idx} | {pred_text} | {lbl} ")
        f1, em = eval(pred_text, lbl)
        f1_scores.append(f1)
        em_scores.append(em)

    print(f"F1 Score : {np.mean(f1_scores)}")
    print(f"EM Score : {np.mean(em_scores)}")