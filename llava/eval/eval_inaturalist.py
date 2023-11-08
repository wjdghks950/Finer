import re
import os
import numpy as np
import json
import string
import argparse

from collections import Counter


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
    if normalize_answer(prediction) in normalize_answer(ground_truth) or \
       normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1.0
    else:
        return (normalize_answer(prediction) == normalize_answer(ground_truth))


def eval(pred, gold):
    f1, precision, recall = f1_score(pred, gold)
    em = exact_match_score(pred, gold)
    return f1, em


def remove_ans_prefix(text, prefix="Answer:"):
    ans_start_idx = text.find(prefix)
    start_idx = 0 if ans_start_idx == -1 else ans_start_idx + len(prefix)
    ans_text = text[start_idx:].strip().replace("</s>", "")
    return ans_text
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default= "/home/jk100/data/data/inaturalist")
    parser.add_argument("--data_path", type=str, default="val_noplant_64.json")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/home/jk100/code/ecole/gpt_output")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse_grained_lbls_gpt4_val_noplant64.json")
    parser.add_argument("--preds_dir", type=str, default="")

    parser.add_argument("--task_type_id", type=int, required=True)
    parser.add_argument("--prompt_type_id", type=int, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
   
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()

    task_types = ["high_coarse", "coarse", "fine"]
    task_type_id = args.task_type_id

    use_prompt = False
    prompt_types = ["cot_0shot", "cot_fewshot", "attr_seek"]
    prompt_type_id = args.prompt_type_id

    # Fix this to change the evaluation types
    task_type = f"{prompt_types[prompt_type_id]}_{task_types[task_type_id]}" if use_prompt else f"{task_types[task_type_id]}"
    
    data_dir = args.data_dir
    data_path = args.data_path
    coarse_lbl_dir = args.coarse_lbl_dir
    coarse_lbl_file = args.coarse_lbl_file

    if "coarse" in task_type:
        with open(os.path.join(coarse_lbl_dir, coarse_lbl_file), "r") as fp:
            coarse_lbls = json.load(fp)
            fp.close()

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
        if "high_coarse" in task_type:
            lbl = category_dict['supercategory'].lower().strip()
        elif "coarse" in task_type:
            lbl = coarse_lbls[str(idx)]
        elif "fine" in task_type:
            lbl = category_dict['name'].lower().strip()
        elif task_type in ['cot', 'l2m', 'attr_prompt']:
            # TODO: Implement the CoT, Least-to-most, Attribute-Seeking Prompt evaluation
            pass

        pred_text = remove_ans_prefix(pred_text)
        print(f"{idx} | {pred_text} | {lbl} ")
        f1, em = eval(pred_text, lbl)
        f1_scores.append(f1)
        em_scores.append(em)

    print(f"F1 Score : {np.mean(f1_scores)}")
    print(f"EM Score : {np.mean(em_scores)}")
