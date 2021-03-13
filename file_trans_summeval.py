import csv
import json
import os
import re
import sys
from collections import defaultdict
import pandas as pd

import numpy as np
import pickle
import pandas as pd
# from scipy.stats import pearsonr, spearmanr, kendalltau
from wodeutil.serialize import PickleHelper
from wodeutil.nlp.metrics.file_util import load_df, get_files_in_dir
from wodeutil.nlp.metrics.summeval_token import *
from wodeutil.os.FileHelper import make_dirs


def clean_summary(seq, clean_sep=False):
    if seq:
        seq = re.sub(r'\n', '', seq)  # remove newline character
        seq = re.sub(r'\t', '', seq)
        if clean_sep:
            seq = re.sub('<t>', '', seq)
            seq = re.sub('</t>', '', seq)
        seq = seq.strip()
        return seq


#
#
# def peek_scores_file():
#     abs_scores, ext_scores = get_all_scores()
#     print(type(abs_scores))
#
#
# def get_syst_names(scores_names):
#     if scores_names:
#         doc_dict = scores_names[0]
#         syst_summs = doc_dict['system_summaries']
#         syst_names = sorted(syst_summs.keys())
#         for i, name in enumerate(syst_names):
#             syst_names[i] = re.sub('.txt', '', name)
#         print(syst_names)
#         return syst_names
#
#
# def get_syst_csv_names(scores):
#     if scores:
#         doc_dict = scores[0]
#         syst_summs = doc_dict['system_summaries']
#         syst_names = sorted(syst_summs.keys())
#         for i, name in enumerate(syst_names):
#             syst_names[i] = re.sub('.txt', '.csv', name)
#         print(syst_names)
#         return syst_names


def get_expert_annoation_stats(annotations):
    """take the average of the 3 calculations from each expert annotator"""
    if annotations:
        coherence = []
        consistency = []
        fluency = []
        relevance = []
        for annotate in annotations:
            coherence.append(annotate['coherence'])
            consistency.append(annotate['consistency'])
            fluency.append(annotate['fluency'])
            relevance.append(annotate['relevance'])
        if coherence and consistency and fluency and relevance:
            # coh, cons, flu, rel = sum(coherence) / len(coherence), sum(consistency) / len(consistency), sum(
            #     fluency) / len(fluency), sum(
            #     relevance) / len(relevance)
            return [sum(coherence) / len(coherence), sum(consistency) / len(consistency), sum(fluency) / len(
                fluency), sum(relevance) / len(relevance)]
        else:
            return -1


def jsonl_to_csv(jsonl_file):
    """ convert paired human annotations jsonl to a csv"""
    print("Reading the input")
    bad_lines = 0
    if jsonl_file is not None:
        try:
            with open(jsonl_file) as inputf:
                make_dirs(summeval_data_all_dir)
                csv_path = os.path.join(summeval_data_all_dir, summeval_data_all_original_name + ".csv")
                with open(csv_path, 'w') as result:
                    print("-------------- start writing to csv -----------------")
                    print(f"write to {csv_path}")
                    data_writer = csv.writer(result, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    headers = ['line_id', 'model_id', 'decoded', 'reference', 'coherence', 'consistency',
                               'fluency',
                               'relevance', 'filepath', 'id', 'text']
                    data_writer.writerow(headers)
                    row_id = 0
                    for count, line in enumerate(inputf):
                        try:
                            data = json.loads(line)
                            if len(data['decoded']) == 0:
                                print(f"bad line: {data['id']}")
                                bad_lines += 1
                                raise ValueError("data id error!")
                            # summary = data['decoded']
                            # references.append(data['reference'])
                            if data.get("reference", None):
                                reference = data['reference']
                            else:  # there are 10 additional references added, the first is the orginal
                                reference = data["references"][0]
                            # article = data['text']
                            annotation_stats = get_expert_annoation_stats(data['expert_annotations'])
                            if not annotation_stats or annotation_stats == -1:
                                raise ValueError("avg_expert is -1!!!")
                            row_item = [row_id, data['model_id'], data['decoded'], reference, *annotation_stats,
                                        data['filepath'], data['id'], data['text']]
                            data_writer.writerow(row_item)
                            row_id += 1
                        except:
                            bad_lines += 1
                            raise ValueError("error when reading inputf!")
                    print(f"write total rows: {row_id}")
                    print("-------------- finish writing to csv -----------------")
        except Exception as e:
            print("Input did not match required format")
            print(e)
            sys.exit()
        print(f"This many bad lines encountered during loading: {bad_lines}")


def rename_column(name: str):
    sent_mover_prefix = "sentence_movers_"
    if name.startswith(sent_mover_prefix):
        name = re.sub(sent_mover_prefix, '', name)
    return name


def add_scores_to_df_with_lineid(df, fp="", metric_list=[]):
    if fp:
        try:
            with open(fp) as inputf:
                dlist = defaultdict(list)
                ids = df['line_id'].values.tolist()
                for count, line in enumerate(inputf):
                    data = json.loads(line)
                    if count != ids[count]:
                        raise ValueError("id doesn't match")
                    for key, value in data.items():
                        if key == "id":
                            continue
                        if key == "rouge":
                            dlist = add_rouge_scores(dlist=dlist, rouge_values=value)
                        else:
                            dlist[key].append(value)
                for k, v in dlist.items():
                    col = rename_column(k)
                    metric_list.append(col)
                    df[col] = v
                    print(f"add {k} with len {len(v)} to df with col {col}")
            return df.copy()
        except Exception as e:
            print(e)
            sys.exit()


def add_scores_to_df(df, fp="", metric_list=[]):
    if fp:
        try:
            with open(fp) as inputf:
                dlist = defaultdict(list)
                # ids = df['line_id'].values.tolist()
                for count, line in enumerate(inputf):
                    data = json.loads(line)
                    if count != data['id']:
                        raise ValueError("id doesn't match")
                    for key, value in data.items():
                        if key == "id":
                            continue
                        if key == "rouge":
                            dlist = add_rouge_scores(dlist=dlist, rouge_values=value)
                        else:
                            dlist[key].append(value)
                for k, v in dlist.items():
                    col = rename_column(k)
                    metric_list.append(col)
                    df[col] = v
                    print(f"add {k} with len {len(v)} to df with col {col}")
            return df.copy()
        except Exception as e:
            print(e)
            sys.exit()


mover_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/evaluation/summ_eval/output_mover_score.jsonl"


def add_rouge_scores(dlist: defaultdict, rouge_values: dict):
    if rouge_values:
        for k, v in rouge_values.items():
            if k == "id":
                continue
            if not k in rouge_metrics_list:
                continue
            dlist[k].append(v)
        return dlist


def collect_scores(df, scores_dir=None, save_to=""):
    if scores_dir and save_to:
        files = get_files_in_dir(scores_dir)
        if files:
            metric_list = []
            for file in files:
                fp = os.path.join(scores_dir, file)
                df = add_scores_to_df(df, fp, metric_list=metric_list)
            df.to_csv(save_to, index=False)
            print(f"saved succefully, collected {len(files)} files with the following columns: ")
            print(metric_list)


def split_by_model(from_path="", save_dir=""):
    if save_dir and from_path:
        df = load_df(from_path)
        models = set(df['model_id'].tolist())
        if models:
            for idx, model in enumerate(models):
                model_path = os.path.join(save_dir, model + ".csv")
                make_dirs(save_dir)
                df_model = df[df['model_id'] == model]
                df_model.to_csv(model_path, index=False)
            print(f"split these {len(models)} models:")
            print(models)


def do_split_by_model():
    # from_path = summeval_data_with_all_scores_path
    from_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/sumeval_all/summeval_all_scores.csv"
    # save_dir = summeval_data_all_syst_dir
    save_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_original"
    split_by_model(from_path=from_path, save_dir=save_dir)


def example_collect_scores():
    scoresdir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/score_files/tempscore"
    save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/sumeval_all/summeval_all_scores.csv"

    df = load_df(save_to)
    df.sort_values(by='line_id', inplace=True)
    collect_scores(df, scores_dir=scoresdir, save_to=save_to)


def create_abs_ext_from_all_df():
    path = summeval_data_all_path
    abs_path = ""
    ext_path = ""


def do_add_scores_to_df():
    # score_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/score_files/output_rouge_we1.jsonl"
    score_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/score_files/output_rouge_we2.jsonl"
    df = load_df(summeval_data_all_with_s3_path)
    df = add_scores_to_df(df, fp=score_path)
    df.to_csv(summeval_data_all_with_s3_path, index=False)


def do_add_scores_to_df_from_dir():
    """
    collect scores.jsonl from a dir and save the collectd scores to df
    """
    # # ext scores
    # score_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/recalculated_realsumm/json_files/scores/ext"
    # data_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/amr_splitted_measure/realsumm_ext_all_metrics.csv"
    # save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/recalculated_realsumm/realsumm_ext_all.csv"

    # abs
    score_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/recalculated_realsumm/json_files/scores/abs"
    data_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/amr_splitted_measure/realsumm_abs_all_metrics.csv"
    save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/recalculated_realsumm/realsumm_abs_all.csv"

    score_files = get_files_in_dir(score_dir)
    df = load_df(data_path)
    df = df.loc[:, ['doc_id', 'model_id', 'cand', 'ref', 'litepyramid_recall', 'js-2', 'cand_amr', 'ref_amr', *col_sema,
                    *col_smatch]].copy()
    print(f"........ start collecting .............")
    for score_file in score_files:
        if not score_file.endswith('.jsonl'):
            continue
        score_path = os.path.join(score_dir, score_file)
        df = add_scores_to_df(df, fp=score_path)
    print(f"........ finish collecting .............")
    df.to_csv(save_to, index=False)
    print(f"........ save to {save_to} .............")


if __name__ == '__main__':
    # peek_scores_file()
    # bart_to_csv()
    # get_syst_csv_names()
    # clean_split_summaries()
    # print_rouge_example()
    # rouge_example()
    # jsonl_file_path = summeval_annotations_paired_jsonl_path
    # jsonl_to_csv(jsonl_file=jsonl_file_path)
    # scores_dir = score_files_dir
    # df_josonl()
    # example_collect_scores()
    # do_split_by_model()
    # example_collect_scores()
    # do_split_by_model()
    do_add_scores_to_df_from_dir()
