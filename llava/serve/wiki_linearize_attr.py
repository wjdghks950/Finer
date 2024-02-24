import requests
import os
import json
import pickle
import random
import jsonlines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ast
import re
import string
import wikipedia
import argparse
# import seaborn

from tqdm import tqdm
from collections import defaultdict

from llava.eval.evals import calculate_scores, align_score
from llava.eval.eval_fgvc import annotation_loader
from llava.utils import remove_text_after_triple_newline

'''
1. For each concept in a dataset (e.g. iNaturalist), search from Wikipedia a document that corresponds to that concept.
2. Linearize each concept's generated attributes from a model (e.g., LLaVA-1.5, GPT-4V).
3. Extract each concept's descriptive, physical attributes from the Wikipedia document.
3. Compare the linearized attributes against the Wikipedia document attributes 
(or the extracted & linearized attributes if args.use_linearized_text == True)
'''


def linearize_attr(concept_name, attr_list, include_concept_name=True):
    '''
    # Demo case
    s = ["red/orange color", "iridescent reflections", "elongated, cylindrical shape", "segmented body", "presence of setae (bristles) on body"]
    linearize_attr(s)
    '''
    linear_attr_str = ""
    if len(attr_list) >= 1:
        if include_concept_name and len(concept_name) > 0:
            linear_attr_str = f"{concept_name} exhibits "
        else:
            linear_attr_str += ", ".join(attr_list)
    return linear_attr_str


def linearize_attr_wiki(concept_attr_dict, dataset_name, out_path, out_dir):
    '''
    Scrape the Wikipedia document that corresponds to a concept and the concept Description
    '''
    max_num_toks = 256

    # inaturalist, stanford_dogs, cub_200_2011, nabirds
    DESC_HEADER = "== Description =="
    desc_pattern = r"== Description ==([\s\S]*)"  # Regular expression pattern to extract text under '== Description =='
    description_text = re.compile(desc_pattern, re.DOTALL)

    # fgvc_aircrafts, stanford_cars
    DESIGN_HEADER = "== Design =="
    design_pattern = r"== Design ==([\s\S]*)"  # Regular expression pattern to extract text under '== Design =='
    design_text = re.compile(design_pattern, re.DOTALL)

    wiki_attr_dicts = []
    no_binomial_cnt, no_common_cnt = 0, 0
    disambig_err = False

    # For iNaturalist, simply load one of the `parsed-model-text-combined.json` and use their `binomial_wiki_doc` and `common_wiki_doc`
    if dataset_name == 'inaturalist':
        cached_wiki_attr_dicts = []
        if os.path.exists(os.path.join(out_dir, "parsed-llava-7b-text-queried-wiki-out.json")):
            with open(os.path.join(out_dir, "parsed-llava-7b-text-queried-wiki-out.json"), "r") as cached_reader:
                for d in tqdm(cached_reader, \
                            desc=f"Loading cached file for use [{os.path.join(out_dir, 'parsed-llava-7b-text-queried-wiki-out.json')}]"):
                    d = json.loads(d)
                    cached_wiki_attr_dicts.append(d)
    
    stopped_idx = 0
    if os.path.exists(os.path.join(out_dir, out_path)):
        # Load if the stored linearized text already exists
        with open(os.path.join(out_dir, out_path), "r") as fp:
            for d in tqdm(fp, desc=f"Loading from an existing path at [{os.path.join(out_dir, out_path)}]"):
                d = json.loads(d)
                wiki_attr_dicts.append(d)
            stopped_idx = len(wiki_attr_dicts)
            print(f"[ starting from stopped_idx = ({stopped_idx}) ]")

    with jsonlines.open(os.path.join(out_dir, out_path), mode='a') as writer_obj:
        # Query Wikipedia for concept descriptions
        for i, cdict in enumerate(tqdm(concept_attr_dict, desc=f"[ Reading [{dataset_name}] concepts from Wikipedia ]")):
            '''
            concept_dict['id'] = id
            concept_dict['name'] = ctgr
            '''
            if i < stopped_idx:
                continue
            id = cdict['id']
            common_wiki_doc = ""
            binomial_wiki_doc = ""
            linearized_text = ""

            if dataset_name == 'inaturalist':
                # Linearized text (from LLM output)
                if len(cdict["attr_binomial"]["required"]) > 0:
                    linearized_text = linearize_attr(cdict['name'], cdict["attr_binomial"]["required"])
                elif len(cdict["attr_common"]["required"]) > 0:
                    linearized_text = linearize_attr(cdict['common_name'], cdict["attr_common"]["required"])

                if len(cached_wiki_attr_dicts) >= len(concept_attr_dict):
                    binomial_wiki_doc = cached_wiki_attr_dicts[i]['binomial_wiki_doc']
                    common_wiki_doc = cached_wiki_attr_dicts[i]['common_wiki_doc']
                    disambig_err = cached_wiki_attr_dicts[i]['disambig_err']
                else:
                    # Search with the binomial nomenclature first
                    binomial_wiki_out = wikipedia.search(cdict['name'])
                    common_wiki_out = wikipedia.search(cdict['common_name'])

                    if len(binomial_wiki_out) > 0:
                        try:
                            page = wikipedia.page(binomial_wiki_out[0])
                            if DESC_HEADER in page.content:
                                binomial_wiki_doc = description_text.search(page.content).group(1)  # remove_text_after_triple_newline(description_text.search(page.content).group(1))
                                binomial_wiki_doc = " ".join(binomial_wiki_doc.split()[:max_num_toks])
                            else:
                                binomial_wiki_doc = page.content  # == Description == header doesn't exist in Wikipedia document
                        except Exception as e:  # Capture the DisambiguationError from BeautifulSoup
                            binomial_wiki_doc = ""
                            disambig_err = True
                            print("DisambiguationError: BeautifulSoup Disambiguation Error occurred!", e)
                    else:  # No Wikipedia result
                        binomial_wiki_doc = ""
                        no_binomial_cnt += 1
                    
                    if len(common_wiki_out) > 0:
                        try:
                            page = wikipedia.page(common_wiki_out[0])
                            if DESC_HEADER in page.content:
                                common_wiki_doc = description_text.search(page.content).group(1)
                                common_wiki_doc = " ".join(common_wiki_doc.split()[:max_num_toks])
                            else:
                                common_wiki_doc = page.content  # == Description == header doesn't exist in Wikipedia document
                                common_wiki_doc = " ".join(common_wiki_doc.split()[:max_num_toks])
                        except Exception as e:  # Capture the DisambiguationError from BeautifulSoup
                            common_wiki_doc = ""
                            disambig_err = True
                            print("DisambiguationError: BeautifulSoup Disambiguation Error occurred!\n", e)
                    else:  # No Wikipedia result
                        common_wiki_doc = ""
                        no_common_cnt += 1

                new_cdict = {"id": id, 
                            "linearized_text": linearized_text,
                            "binomial_wiki_doc": binomial_wiki_doc,
                            "common_wiki_doc": common_wiki_doc,
                            "disambig_err": disambig_err,
                            }
            else:
                print(f"[ Search Query : {cdict['name'].strip()} ]")
                fine_wiki_out = wikipedia.search(cdict['name'])
                if len(fine_wiki_out) > 0:
                    try:
                        page = wikipedia.page(fine_wiki_out[0])
                        if DESC_HEADER in page.content:
                            common_wiki_doc = description_text.search(page.content).group(1)
                        elif DESIGN_HEADER in page.content:
                            common_wiki_doc = design_text.search(page.content).group(1)
                        else:
                            common_wiki_doc = page.content  # '== Description ==' OR '== Design ==' doesn't exist in Wikipedia document
                    except Exception as e:  # Capture the DisambiguationError from BeautifulSoup
                        disambig_err = True
                        print("DisambiguationError: BeautifulSoup Disambiguation Error occurred!", e)
                        no_common_cnt += 1
                    common_wiki_doc = " ".join(common_wiki_doc.split()[:max_num_toks])

            new_cdict = {"id": id, 
                        "query": cdict['name'],
                        "common_wiki_doc": common_wiki_doc,
                        "disambig_err": disambig_err,
                        }
            writer_obj.write(new_cdict)
            wiki_attr_dicts.append(new_cdict)
            disambig_err = False

        if i + 1 % 500:
            print(new_cdict)
            print("no_binomial_cnt: ", no_binomial_cnt)
            print("no_common_cnt: ", no_common_cnt)
        
        writer_obj.close()

    return wiki_attr_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="inaturalist")
    parser.add_argument("--data-dir", type=str, default="/home/jk100/data/data/inaturalist")
    parser.add_argument("--img-dir", type=str, default="val")
    parser.add_argument("--val-data-path", type=str, default="val.json")
    parser.add_argument("--train-data-path", type=str, default="train.json")
    parser.add_argument("--preds_dir", type=str, default="/home/jk100/code/ecole/LLaVA/llava/preds")
    parser.add_argument("--eval-out-path", type=str, default="EVAL_RESULT")

    parser.add_argument("--common-attr-path", type=str, default="parsed-llama-70b-common-names.json", help='combining required, likely attributes')
    parser.add_argument("--binomial-attr-path", type=str, default="parsed-llama-70b-species-names.json")
    parser.add_argument("--combined-attr-path", type=str, default="parsed-llama-70b-combined.json")
    parser.add_argument("--train-data-file", type=str, default="train.json")
    parser.add_argument("--concept-wiki-path", type=str, default="parsed-llama-70b-queried-wiki-out.json")
    parser.add_argument("--use-linearized-text", action="store_true", help="Use linearized_text as reference text")

    parser.add_argument("--data_path", type=str, default="val_noplant_64.json")
    parser.add_argument("--coarse_lbl_dir", type=str, default= "/home/jk100/code/ecole/gpt_output")
    parser.add_argument("--coarse_lbl_file", type=str, default="coarse_grained_lbls_gpt4_val_noplant64.json")
    parser.add_argument("--ctgr2img-path", type=str, default="ctgr2img_dict.json")

    args = parser.parse_args()


    # TODO: Implement a Wikipedia API query script for all the datasets
    # TODO: Write a shell script that enables that
    # TODO: Unify any dangling methods


    # Retrieved datasets
    # dataset_name = "inaturalist"  # TODO: Replace with 'args.dataset_name'
    dataset_name = args.dataset_name
    preds_dir = args.preds_dir

    train_annot_path = args.train_data_file

    if dataset_name == "inaturalist":
        test_annot_path = "val.json" # "unified-inaturalist-test-combined.jsonl"
        train_annot_path = "train.json" # "unified-inaturalist-train-combined.jsonl"
    elif dataset_name == "fgvc_aircraft":
        test_annot_path = "unified-fgvc-aircraft-test-combined.jsonl"
        train_annot_path = "unified-fgvc-aircraft-train-combined.jsonl"
        dataset_name = "fgvc-aircraft-2013b"
    elif dataset_name == "stanford_dogs":
        test_annot_path = "unified-stanford-dogs-test-combined.jsonl"
        train_annot_path = "unified-stanford-dogs-train-combined.jsonl"
    elif dataset_name == "cub_200_2011":
        test_annot_path = "unified-cub-200-test-combined.jsonl"
        train_annot_path = "unified-cub-200-train-combined.jsonl"
        dataset_name = "CUB_200_2011"
    elif dataset_name == "nabirds":
        test_annot_path = "unified-nabirds-test-combined.jsonl"
        train_annot_path = "unified-nabirds-train-combined.jsonl"
    elif dataset_name == "stanford_cars":
        test_annot_path = "unified-stanford-cars-test-combined.jsonl"
        train_annot_path = "unified-stanford-cars-train-combined.jsonl"

    data_dir = f"/home/jk100/data/data/{dataset_name}"

    common_attr_path = args.common_attr_path  # Attributes extracted by their common names
    binomial_attr_path = args.binomial_attr_path  # Attributes extracted by their binomial nomenclature

    # Load the training dataset & common_name, species attributes extracted from LLAMA-70B
    if dataset_name == 'inaturalist':
        if os.path.exists(os.path.join(data_dir, train_annot_path)):
            with open(os.path.join(data_dir, train_annot_path)) as reader_obj:
                train_dict = json.load(reader_obj)
                print("Keys in the train annotation : ", train_dict.keys())
        else:
            with jsonlines.open(os.path.join(data_dir, train_annot_path)) as reader_obj:
                train_dict = [obj for obj in reader_obj]

        if os.path.exists(os.path.join(data_dir, common_attr_path)):
            with open(os.path.join(data_dir, common_attr_path)) as reader_obj:
                common_attr_dict = json.load(reader_obj)
                print(f"Common Name (attr_dict size) : ", len(common_attr_dict))

        if os.path.exists(os.path.join(data_dir, binomial_attr_path)):
            with open(os.path.join(data_dir, binomial_attr_path)) as reader_obj:
                binomial_attr_dict = json.load(reader_obj)
                print(f"Species Name (attr_dict size) : ", len(binomial_attr_dict))

        combined_path = args.combined_attr_path
        if not os.path.exists(os.path.join(data_dir, combined_path)):
            # Convert `concept names` (common_name and name) to keys and `attributes` to values
            common_attrs = {c['concept']: {'required': c['required'], 'likely': c['likely']} for c in common_attr_dict}
            binomial_attrs = {c['concept']: {'required': c['required'], 'likely': c['likely']} for c in binomial_attr_dict}
            
            # Construct a common dict with overlapping concepts
            concept_attr_dict = []
            for ctgr in tqdm(train_dict['categories'], desc="Constructing a joint concept_dict"):
                concept_dict = {}
                concept_dict['id'] = ctgr['id']
                concept_dict['name'] = ctgr['name']
                concept_dict['common_name'] = ctgr['common_name']

                # Attributes generated using common_name
                try:
                    concept_dict['attr_common'] = common_attrs[ctgr['common_name']]
                except KeyError:
                    concept_dict['attr_common'] = {'required': [], 'likely': []}
                # Attributes generated using name (i.e., binomial nomenclature)
                try:
                    concept_dict['attr_binomial'] = binomial_attrs[ctgr['name']]
                except KeyError:
                    concept_dict['attr_binomial'] = {'required': [], 'likely': []}
                concept_attr_dict.append(concept_dict)
            with open(os.path.join(data_dir, combined_path), "w") as writer_obj:
                json.dump(concept_attr_dict, writer_obj)
                print(f"{combined_path} saved in {os.path.join(data_dir, combined_path)}")

        else:
            with open(os.path.join(data_dir, combined_path), "r") as reader_obj:
                concept_attr_dict = json.load(reader_obj)
                print(f"Loading from {os.path.join(data_dir, combined_path)}")

    else:  # Datasets other than inaturalist
        # Loading the ground-truth labels for each dataset from 'test_annot_path'
        ground_truths = annotation_loader(os.path.join(data_dir, test_annot_path))
        concept_attr_dict = []
        fine_lbls = list(set([lbl.strip() for obj in ground_truths for lbl in obj['fine-level-lbl']]))  # Construct a ground-truth concept label set for fine-grained level concepts
        print(f"[ Number of fine-grained concepts : {len(fine_lbls)} ]")
        print("fine_lbls : ", fine_lbls)
        for id, ctgr in enumerate(tqdm(fine_lbls, desc=f"Constructing a joint concept_dict from [{dataset_name}]")):
            concept_dict = {}
            concept_dict['id'] = id
            concept_dict['name'] = ctgr
            concept_attr_dict.append(concept_dict)

    # Retrieve Wikipedia documents for each concept and linearize LLama-70B-generated attributes and save them to `concept_wiki_path`
    if dataset_name == 'inaturalist':
        if "gpt-4-wiki-text" in combined_path:
            concept_wiki_path = "parsed-" + "-".join(combined_path.split("-")[1:5]) + "-queried-wiki-out.json"
        else:
            concept_wiki_path = "parsed-" + "-".join(combined_path.split("-")[1:4]) + "-queried-wiki-out.json"
    else:
        concept_wiki_path = "parsed-" + f"{dataset_name}" + "-queried-wiki-out.json"
    print(f"OUT_PATH - concept_wiki_path : [{concept_wiki_path}]")

    # Retrieve Wikipedia documents and save them along with the linearized attributes text
    out_dir = os.path.join(preds_dir, f"parsed_{dataset_name}_outputs")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    wiki_attr_dicts = linearize_attr_wiki(concept_attr_dict, dataset_name, concept_wiki_path, out_dir=out_dir)

    exit()
    if args.use_linearized_text:  # TODO: Refactor this part so it incorporates other datasets as well - merge with calculate_score.py
        # Attributes extracted from Wikipedia using GPT-4 
        # (the resulting `linearized_text` is used as reference text for evaluation)
        ref_path = "parsed-gpt-4-wiki-text-queried-wiki-out.json"
        gpt4_wiki_attr_dicts = []
        print(f"Using linearized attributes extracted from Wikipedia by GPT-4 ...")
        with open(os.path.join(data_dir, ref_path), "r") as reader_obj:
            for d in reader_obj:
                gpt4_wiki_attr_dicts.append(json.loads(d))
            reader_obj.close()


    eval_out_path = args.eval_out_path
    if not os.path.exists(eval_out_path):
        os.mkdir(eval_out_path)

    if args.use_linearized_text:
        score_out_path = os.path.join(eval_out_path, "_".join(concept_wiki_path.split("-")[1:4]) + "_wiki") + ".jsonl"
    else:
        score_out_path = os.path.join(eval_out_path, "_".join(concept_wiki_path.split("-")[1:4])) + ".jsonl"


    demo_idx = 20  # FIXME: Erase if done with demo evaluation
    
    scores = []
    if os.path.exists(score_out_path):
        with open(score_out_path, "r") as reader_obj:
            for score_dict in tqdm(reader_obj, desc=f"Loading from the existing scores file : {score_out_path}"):
                scores.append(json.loads(score_dict))
            reader_obj.close()

    # Calculate the n-gram overlap (e.g., BLEU, ROUGE) and BERTscore against Wikipedia text and generated texts
    no_txt_cnt, no_ref_cnt = 0, 0
    with jsonlines.open(score_out_path, "a") as writer_obj:
        for i, cdict in enumerate(tqdm(wiki_attr_dicts[:demo_idx], desc=f"=== Calculating BLEU, ROUGE and BERTScore for generated attributes from {concept_wiki_path} ===\n")):
            if i < len(scores):
                continue
            print(f"[ IDX : {i} ]\n")
            if args.use_linearized_text:  # Using attributes extracted & linearized from Wikipedia using GPT-4 
                reference = gpt4_wiki_attr_dicts[i]['linearized_text']
            else:
                if cdict['binomial_wiki_doc'] == "" and cdict['common_wiki_doc'] == "":  # Using the entire Wikipedia doc as reference text
                    no_ref_cnt += 1
                    continue
                else:
                    reference = cdict['binomial_wiki_doc'] if cdict['binomial_wiki_doc'] != "" else cdict['common_wiki_doc']
            
            scores.append(calculate_scores(cdict['linearized_text'], reference))
            writer_obj.write(calculate_scores(cdict['linearized_text'], reference))
        
        print("no_txt_cnt : ", no_txt_cnt)
        print("no_reference_cnt : ", no_ref_cnt)
        writer_obj.close()

    # TODO: Add AlignScore here later from `calculate_alignscore_inaturalist.py`
    '''
    {"BLEU": 3.967454049950442e-234, 
    "ROUGE": {"rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0}, "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0}, 
    "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0}}, 
    "BERTScore": {"precision": 0.879917562007904, "recall": 0.8043099045753479, "f1": 0.840416669845581}}
    '''
    bleu_out = 0.0
    rouge_1, rouge_2, rouge_l = 0.0, 0.0, 0.0
    bert_score_prec, bert_score_rec, bert_score_f1 = 0.0, 0.0, 0.0
    for s in scores:
        bleu_out += s['BLEU']
        rouge_1 += s['ROUGE']['rouge-1']['f']
        rouge_2 += s['ROUGE']['rouge-2']['f']
        rouge_l += s['ROUGE']['rouge-l']['f']
        bert_score_f1 += s['BERTScore']['f1']

    print("="*20)
    print(f"Evaluated Model Output Path: [ {concept_wiki_path} ]\n")
    print(f"BLEU = {bleu_out / len(scores)}")
    print(f"ROUGE-1 = {rouge_1 / len(scores)}")
    print(f"ROUGE-2 = {rouge_2 / len(scores)}")
    print(f"ROUGE-L = {rouge_l / len(scores)}")
    print(f"BERTScore = {bert_score_f1 / len(scores)}")
    print("="*20)
    