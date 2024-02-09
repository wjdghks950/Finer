import re
import os
import numpy as np
import json
import jsonlines
import string
import argparse

from collections import Counter
from tqdm import tqdm


def annotation_loader(path):
    if os.path.exists(path):
        with jsonlines.open(path, "r") as reader:
            ground_truths = [d for d in reader]
            print(f"Length of the annotated file (data instance #): {len(ground_truths)}")
    else:
        raise FileNotFoundError(f"{path} does not exist!")
    return ground_truths


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
    normalized_ground_truth = " ".join(normalize_answer(ground_truth).split()[:10])  # Calculate only till the first 10 words to prevent over-generation by smaller models like 7B

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


def exact_match_score(prediction, ground_truth, max_tok_cnt=20):
    if normalize_answer(ground_truth) in " ".join(normalize_answer(prediction).split()[:max_tok_cnt]):
        # Modified EM - if the ground_truth is within the first 20 generated words, give it 1.0
        return 1.0
    else:
        return int(normalize_answer(prediction) == normalize_answer(ground_truth))


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
    parser.add_argument("--data_path", type=str, default="val.json")
    parser.add_argument("--dataset_name", type=str, default="inaturalist")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/home/jk100/code/ecole/gpt_output")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse_grained_lbls_gpt4_val_noplant64.json")
    parser.add_argument("--preds_dir", type=str, default="/shared/nas/data/m1/jk100/code/ecole/LLaVA/llava/preds")
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
   
    args = parser.parse_args()

    out_dir = os.path.join(args.preds_dir, args.dataset_name + "_outputs")
    print("OUT_DIR : ", out_dir)
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"{out_dir} does not exist!")

    if args.dataset_name == "inaturalist":
        annot_path = "unified-inaturalist-test-combined.jsonl"
    elif args.dataset_name == "fgvc_aircraft":
        annot_path = "unified-fgvc-aircraft-test-combined.jsonl"
    elif args.dataset_name == "stanford_dogs":
        annot_path = "unified-stanford-dogs-test-combined.jsonl"
    elif args.dataset_name == "cub_200_2011":
        annot_path = "unified-cub-200-test-combined.jsonl"
    elif args.dataset_name == "nabirds":
        annot_path = "unified-nabirds-test-combined.jsonl"
    elif args.dataset_name == "stanford_cars":
        annot_path = "unified-stanford-cars-test-combined.jsonl"

    # Loading the ground-truth labels for each dataset from 'annot_path
    ground_truths = annotation_loader(os.path.join(args.data_dir, annot_path))

    task_types = ["high_coarse", "coarse", "fine"]

    # TODO: Erase use_prompt
    # use_prompt = args.use_prompt
    # prompt_types = ["cot_0shot", "cot_fewshot", "attr_seek"]
    # prompt_type_id = args.prompt_type_id

    # Paths to VLM-generated prediction outputs
    pred_dir = os.path.join(args.preds_dir, args.dataset_name + "_outputs")
    pred_paths = []
    for tidx, task_type in enumerate(task_types):
        out_path = f"{args.dataset_name}_{task_type}_{args.model_name}_output.jsonl"
        pred_paths.append((tidx, out_path))

    pred_dict = {task_type: [] for task_type in task_types}
    for tidx, pred_path in pred_paths:
        task_type = task_types[tidx]
        try:
            with jsonlines.open(os.path.join(pred_dir, pred_path), "r") as reader:
                pred_dict[task_type] = [line for line in reader]
                print(f"({pred_path}) => LENGTH: ", len(pred_dict[task_type]))
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            print(f"[ Skipping : {pred_path} ]")
            pred_dict[task_type] = [{"idx": 0, "text": ""}]

    score_dict = {task_type: {'em': [], 'f1': []} for task_type in task_types}

    for task_type in tqdm(task_types, desc=f"[ Evaluating: {args.dataset_name} ]"):
        print(f"\n\n==TASK: {task_type}==")
        if task_type == "high_coarse":
            lbl_key = "basic-level-lbl"
        elif task_type == "coarse":
            lbl_key = "coarse-level-lbl"
        elif task_type == "fine":
            lbl_key = "fine-level-lbl"

        for pdict in tqdm(pred_dict[task_type]):
            idx = int(pdict["idx"])
            output_text = remove_ans_prefix(pdict["text"])  # Model-generated text

            ems, f1s = [], []  # Considers maximum EM and F1 scores for high_coarse and coarse cases
            ground_truth_lbls = []
            if task_type == "high_coarse":
                # For 'basic-level' case, if the models generate either one of the three granularity labels, consider them correct
                ground_truth_lbls += [ground_truths[idx]["basic-level-lbl"]] + ground_truths[idx]["coarse-level-lbl"] + ground_truths[idx]["fine-level-lbl"]
            elif task_type == "coarse":
                # For 'coarse-grained' case, if the models generate fine-level-lbl, consider them correct as well
                ground_truth_lbls += ground_truths[idx]["coarse-level-lbl"] + ground_truths[idx]["fine-level-lbl"]
            elif task_type == "fine":
                ground_truth_lbls += ground_truths[idx]["fine-level-lbl"]

            # print(f"[ ({idx}) GROUND_TRUTH vs. OUTPUT_TEXT: {ground_truth_lbls} || {normalize_answer(output_text)}")

            for lbl in ground_truth_lbls:
                temp_f1, temp_em = eval(output_text, lbl)
                f1s.append(temp_f1)
                ems.append(temp_em)
            em = max(ems)
            f1 = max(f1s)
            score_dict[task_type]['em'].append(em)
            score_dict[task_type]['f1'].append(f1)

    for task_type in task_types:
        print(f"\n[ ====== {task_type} ====== ]")
        print(f"F1 Score : {np.mean(score_dict[task_type]['f1'])}")
        print(f"EM Score : {np.mean(score_dict[task_type]['em'])}")
