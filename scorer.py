import os

import spacy
from summ_eval.rouge_metric import RougeMetric
from summ_eval.rouge_we_metric import RougeWeMetric
from summ_eval.bert_score_metric import BertScoreMetric
from summ_eval.blanc_metric import BlancMetric
from summ_eval.chrfpp_metric import ChrfppMetric
from summ_eval.cider_metric import CiderMetric
from summ_eval.mover_score_metric import MoverScoreMetric
from summ_eval.sentence_movers_metric import SentenceMoversMetric
from summ_eval.summa_qa_metric import SummaQAMetric
from summ_eval.meteor_metric import MeteorMetric
from summ_eval.bleu_metric import BleuMetric
from wodeutil.nlp.metrics.file_util import load_df, get_files_in_dir, clean_cnn_text
from collections import defaultdict
from wodeutil.nlp.metrics.token import realsumm_data_abs_all_path, realsumm_data_abs_all_extra_path
from summ_eval.data_stats_metric import DataStatsMetric
from summ_eval.s3_metric import S3Metric
import pandas as pd
from summ_eval.test_util import CAND_R, REF_R, rouge_output, rouge_output_batch, CANDS, REFS, EPS

from set_path import set_path


# ROUGE_HOME = os.environ['ROUGE_HOME']


class MetricScorer():
    def __init__(self, ROUGE_HOME=None):
        if ROUGE_HOME:
            self.ROUGE_HOME = ROUGE_HOME
        else:
            self.ROUGE_HOME = os.environ['ROUGE_HOME']
        self.rouge = None
        self.bertscorer = None
        self.chrfpp = None
        self.cider = None
        self.moverscore = None
        self.sent_mover = None
        self.summaqa = None
        self.meteor = None
        self.bleu = None
        self.blanc = None
        self.txt_stats = None
        self.s3 = None

    def get_rouge(self, cand, refs, batch_mode=False):
        """
        cand: str
        refs: str or list of strings
        """
        if not self.rouge:
            self.rouge = RougeMetric(self.ROUGE_HOME)
        if cand and refs:
            if batch_mode:
                score_dict = self.rouge.evaluate_batch(cand, refs)["rouge"]
            else:
                score_dict = self.rouge.evaluate_example(cand, refs)['rouge']
            return score_dict

    def get_rouge_we(self, cands, refs, n_gram=1, batch_mode=False, aggregate=True):
        if cands and refs:
            self.rouge_we = RougeWeMetric(n_gram=n_gram)
            if batch_mode:
                score = self.rouge_we.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.rouge_we.evaluate_example(cands, refs)
            return score

    def get_bert_score(self, cands, refs, batch_mode=False, aggregate=True, use_default_params=True, lang='en',
                       model_type='roberta-large', num_layers=17,
                       verbose=False, idf=False, batch_size=3, rescale_with_baseline=False):
        """
        cands: str
        refs: list of strings
        """
        if cands and refs:
            bertscorer = self.bertscorer
            if use_default_params:
                if not bertscorer:
                    self.bertscorer = BertScoreMetric(lang='en', model_type='roberta-large', num_layers=17,
                                                      verbose=False, idf=False, \
                                                      batch_size=3, rescale_with_baseline=False)
                    bertscorer = self.bertscorer
            else:
                bertscorer = BertScoreMetric(lang=lang, model_type=model_type, num_layers=num_layers,
                                             verbose=verbose, idf=idf, \
                                             batch_size=batch_size, rescale_with_baseline=rescale_with_baseline)
            if batch_mode:
                score_dict = bertscorer.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score_dict = bertscorer.evaluate_example(cands, refs)
            return score_dict

    def get_chrfpp(self, cands, refs, batch_mode=False, aggregate=True):
        if not self.chrfpp:
            self.chrfpp = ChrfppMetric()
        if cands and refs:
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.chrfpp.evaluate_batch(cands, refs,
                                                   aggregate=aggregate)  # not working due to ziplongest method, need to fix later
            else:
                score = self.chrfpp.evaluate_example(cands, refs)
            return score

    def get_cider(self, cands, refs, tokenize=False, batch_mode=True, aggregate=True):
        if cands and refs:
            if not self.cider:
                self.cider = CiderMetric(tokenize=tokenize)
            if batch_mode:
                score = self.cider.evaluate_batch(cands, refs, aggregate=aggregate)
            # else:
            #     score = self.cider.evaluate_example(cands, refs)
            return score

    def str_to_list(self, cands, refs, exclude_cands=False, exclude_refs=False):
        """
        if cands or refs is of type str, turn them into list
        """
        if cands and refs:
            if isinstance(cands, str) and not exclude_cands:
                cands = [cands]
            if isinstance(refs, str) and not exclude_refs:
                refs = [refs]
            return cands, refs

    def get_moverscore(self, cands, refs, version=2, use_default=True, stop_wordsf=None, n_gram=1, remove_subwords=True,
                       aggregate=False, batch_mode=False):
        """
        default using version 2
        """
        if cands and refs:
            if use_default:
                if not self.moverscore:
                    self.moverscore = MoverScoreMetric(version=version, stop_wordsf=stop_wordsf, n_gram=n_gram,
                                                       remove_subwords=remove_subwords)
                moverscore = self.moverscore
            else:
                moverscore = MoverScoreMetric(version=version, stop_wordsf=stop_wordsf, n_gram=n_gram,
                                              remove_subwords=remove_subwords)

            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                scores = moverscore.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                scores = moverscore.evaluate_example(cands, refs)
            return scores

    def get_txt_stats(self, cand, ref):
        if cand and ref:
            if not self.txt_stats:
                self.txt_stats = DataStatsMetric()
            stats = self.txt_stats.evaluate_example(cand, ref)
            return stats

    def get_s3(self, cand, ref, batch_mode=False):
        if cand and ref:
            if not self.s3:
                self.s3 = S3Metric()
            if batch_mode:
                score = self.s3.evaluate_batch(cand, ref)
            else:
                score = self.s3.evaluate_example(cand, ref)
            return score

    def get_JS(self, cand, ref, batch_mode=False):
        if cand and ref:
            if not self.s3:
                self.s3 = S3Metric()
            if batch_mode:
                score = self.s3.evaluate_JS_batch(cand, ref)
            else:
                score = self.s3.get_JS(cand, ref)
            return score

    def get_sent_mover(self, cands, refs, wordrep='glove', metric_type='sms', batch_mode=False):
        """
        metric_type: sms, wms, s+wms
        """
        if cands and refs:
            if not self.sent_mover:
                self.sent_mover = SentenceMoversMetric(wordrep=wordrep, metric=metric_type)
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.sent_mover.evaluate_batch(cands, refs)
            else:
                score = self.sent_mover.evaluate_example(cands, refs)
            return score

    def get_summaqa(self, cands, src_article, use_default_batch=True, batch_size=8, batch_mode=False):
        if cands and src_article:
            if use_default_batch:
                if not self.summaqa:
                    self.summaqa = SummaQAMetric(batch_size=batch_size)
                    summaqa = self.summaqa
                else:
                    summaqa = SummaQAMetric(batch_size=batch_size)
            if batch_mode:
                cands, refs = self.str_to_list(cands, src_article)
                score = summaqa.evaluate_batch(cands, src_article)
            else:
                score = summaqa.evaluate_example(cands, src_article)
            return score

    def get_meteor(self, cands, refs, batch_mode=False, aggregate=True):
        if cands and refs:
            if not self.meteor:
                self.meteor = MeteorMetric()
            if batch_mode:
                cands, refs = self.str_to_list(cands, refs)
                score = self.meteor.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.meteor.evaluate_example(cands, refs)
            return score

    def get_bleu(self, cands, refs, batch_mode=False, aggregate=True):
        if cands and refs:
            if not self.bleu:
                self.bleu = BleuMetric()
            if batch_mode:
                score = self.bleu.evaluate_batch(cands, refs, aggregate=aggregate)
            else:
                score = self.bleu.evaluate_example(cands, refs)
            return score

    def get_blanc(self, cands, src_texts, use_tune=True, use_default=True, batch_mode=False, aggregate=True,
                  device="cpu"):
        if cands and src_texts:
            if use_default:
                if not self.blanc:
                    blanc = self.blanc = BlancMetric(use_tune=use_tune, device=device)
            else:
                blanc = BlancMetric(use_tune=use_tune, device=device)
            if batch_mode:
                score = blanc.evaluate_batch(cands, src_texts, aggregate=aggregate)
            else:
                score = blanc.evaluate_example(cands, src_texts)
            return score

    def add_metric_score(self, score_dict, results: defaultdict, is_batch_mode=False):
        if score_dict:
            if isinstance(score_dict, dict):
                for k, v in score_dict.items():
                    results[k].append(v)
            elif isinstance(score_dict, list):
                for score in score_dict:
                    for k, v in score.items():
                        results[k].append(v)

    def add_JS_to_results(self, cand: str, ref: str, results: defaultdict):
        if cand and ref:
            js_dict = scorer.get_JS(cand, ref, batch_mode=True)
            self.add_metric_score(js_dict, results=results)

    def add_scores_to_results(self, cand: str, ref: str, results: defaultdict, include_rouge=False,
                              include_bertscore=True, include_sent_mover=False, include_rwe=True, \
                              include_moverscore=False, include_bleu=True, include_meteor=True, include_txt_stats=True):
        if cand and ref:
            if include_rouge:
                rouge_dict = self.get_rouge(cand, ref)
                self.add_metric_score(rouge_dict, results=results)
            # rouge_we
            if include_rwe:
                rwe1 = self.get_rouge_we(cand, ref, n_gram=1)
                rwe2 = self.get_rouge_we(cand, ref, n_gram=2)
                self.add_metric_score([rwe1, rwe2], results=results)
            # bertscore
            if include_bertscore:
                bertscore = self.get_bert_score(cand, ref)
                self.add_metric_score(bertscore, results=results)
            # moverscore
            if include_moverscore:
                moverscore = self.get_moverscore(cand, ref)  # defalut version 2
                self.add_metric_score(moverscore, results=results)
            # sent_mover
            if include_sent_mover:
                sent_mover = self.get_sent_mover(cand, ref)
                self.add_metric_score(sent_mover, results=results)
            # bleu
            if include_bleu:
                bleu_score = self.get_bleu(cand, ref)
                self.add_metric_score(bleu_score, results=results)
            # meteor
            if include_meteor:
                meteor_score = self.get_meteor(cand, ref)
                self.add_metric_score(meteor_score, results=results)
            if include_txt_stats:
                txt_stats = self.get_txt_stats(cand, ref)
                self.add_metric_score(txt_stats, results=results)

    def add_batch_scores_to_results(self, cand: str, ref: str, results: defaultdict, include_rouge=True,
                                    include_bertscore=True, include_sent_mover=True, include_rwe=True, \
                                    include_moverscore=True, include_bleu=True, include_meteor=True,
                                    include_txt_stats=True):
        if cand and ref:
            if include_rouge:
                rouge_dict = self.get_rouge(cand, ref, batch_mode=True)
                self.add_metric_score(rouge_dict, results=results)
            # rouge_we
            if include_rwe:
                rwe1 = self.get_rouge_we(cand, ref, n_gram=1, batch_mode=True, aggregate=False)
                rwe2 = self.get_rouge_we(cand, ref, n_gram=2, batch_mode=True, aggregate=False)
                self.add_metric_score(rwe1, results=results)
                self.add_metric_score(rwe2, results=results)

            # bertscore
            if include_bertscore:
                bertscore = self.get_bert_score(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(bertscore, results=results)
            # moverscore
            if include_moverscore:
                moverscore = self.get_moverscore(cand, ref, batch_mode=True, aggregate=False)  # defalut version 2
                self.add_metric_score(moverscore, results=results)
            # sent_mover
            if include_sent_mover:
                sent_mover = self.get_sent_mover(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(sent_mover, results=results)
            # bleu
            if include_bleu:
                bleu_score = self.get_bleu(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(bleu_score, results=results)
            # meteor
            if include_meteor:
                meteor_score = self.get_meteor(cand, ref, batch_mode=True, aggregate=False)
                self.add_metric_score(meteor_score, results=results)
            # if include_txt_stats:
            #     txt_stats = self.get_txt_stats(cand, ref)
            #     self.add_metric_score(txt_stats, results=results)

    def cal_batch_scores_from_txt(self, cands, refs, save_to=""):
        if cands and refs:
            results = defaultdict(list)
            assert len(cands) == len(refs)
            "cands and refs should be of the same length"
            print(f"................ start calculating batch scores for {len(cands)} cand ref pair(s) ................")
            self.add_batch_scores_to_results(cands, refs, results, include_rouge=False, include_moverscore=False,
                                             include_bertscore=False,
                                             include_sent_mover=False, include_rwe=False)
            # for cand, ref in zip(cands, refs):
            #     self.add_scores_to_results(cand, ref, results)
            df = pd.DataFrame(results)
            # for k, v in results.items():
            #     df[k] = v
            print(
                f"................ finish calculating  batch scores for {len(cands)} cand ref pair(s) ................")
            if save_to:
                df.to_csv(save_to, index=False)
                print(
                    f"................ append batch results and save it to {save_to} ...............")

    def cal_scores_from_csv(self, src_path="", save_to="", clean_text=False, cand_col="cand", ref_col="ref"):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if src_path:
            df = load_df(src_path)[:20].copy()
            cands = df[cand_col].values.tolist()
            refs = df[ref_col].values.tolist()
            if cands and refs:
                results = defaultdict(list)
                assert len(cands) == len(refs)
                "cands and refs should be of the same length"
                print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
                for cand, ref in zip(cands, refs):
                    if clean_text:
                        cand = clean_cnn_text(cand, remove_newline=True, clean_sep=True)
                        ref = clean_cnn_text(ref, remove_newline=True, clean_sep=True)
                    self.add_scores_to_results(cand, ref, results)
                    self.add_JS_to_results(cand, ref, results=results)
                for k, v in results.items():
                    df[k] = v
                print(f"................ finish calculating scores for {len(cands)} cand ref pair(s) ................")
                if save_to:
                    df.to_csv(save_to)
                    print(
                        f"................ append results and save it to {save_to} ...............")

    def cal_JS_scores_from_csv(self, src_path="", save_to="", clean_text=False, cand_col="cand", ref_col="ref"):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if src_path:
            df = load_df(src_path)
            cands = df[cand_col].values.tolist()
            refs = df[ref_col].values.tolist()
            if cands and refs:
                results = defaultdict(list)
                assert len(cands) == len(refs)
                "cands and refs should be of the same length"
                print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
                for cand, ref in zip(cands, refs):
                    if clean_text:
                        cand = clean_cnn_text(cand, remove_newline=True, clean_sep=True)
                        ref = clean_cnn_text(ref, remove_newline=True, clean_sep=True)
                    self.add_JS_to_results(cand, ref, results=results)
                for k, v in results.items():
                    df[k] = v
                print(f"................ finish calculating scores for {len(cands)} cand ref pair(s) ................")
                if save_to:
                    df.to_csv(save_to)
                    print(
                        f"................ append results and save it to {save_to} ...............")

    def cal_JS_scores_from_df(self, df, save_to="", clean_text=False, cand_col="cand", ref_col="ref"):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if not df is None:
            cands = df[cand_col].values.tolist()
            refs = df[ref_col].values.tolist()
            if cands and refs:
                results = defaultdict(list)
                assert len(cands) == len(refs)
                "cands and refs should be of the same length"
                print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
                for cand, ref in zip(cands, refs):
                    if clean_text:
                        cand = clean_cnn_text(cand, remove_newline=True, clean_sep=True)
                        ref = clean_cnn_text(ref, remove_newline=True, clean_sep=True)
                    self.add_JS_to_results(cand, ref, results=results)
                for k, v in results.items():
                    df[k] = v
                print(f"................ finish calculating scores for {len(cands)} cand ref pair(s) ................")
                if save_to:
                    df.to_csv(save_to)
                    print(
                        f"................ append results and save it to {save_to} ...............")

    def cal_scores_from_txt_for_realsumm(self, cands="", refs="", clean_text=False):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if cands and refs:
            results = defaultdict(list)
            assert len(cands) == len(refs)
            "cands and refs should be of the same length"
            print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
            self.cal_batch_scores_from_txt(cands, refs)

    def cal_scores_from_csv_for_realsumm(self, src_path="", save_to="", clean_text=True):
        """
        return a dataframe contains the calculated scores
        save to the save_to path if it is specified
        assume the cand summary is under column 'cand' and reference summary is under column 'ref'
        """
        if src_path:
            df = load_df(src_path)
            df = df.loc[:3, :].copy()
            cands = df['cand'].values.tolist()
            refs = df['ref'].values.tolist()
            if cands and refs:
                results = defaultdict(list)
                assert len(cands) == len(refs)
                "cands and refs should be of the same length"
                print(f"................ start calculating scores for {len(cands)} cand ref pair(s) ................")
                for cand, ref in zip(cands, refs):
                    if clean_text:
                        cand = clean_cnn_text(cand, remove_newline=True, clean_sep=True)
                        ref = clean_cnn_text(ref, remove_newline=True, clean_sep=True)
                    self.add_scores_to_results(cand, ref, results, include_rouge=False, include_rwe=True,
                                               include_bertscore=True, include_moverscore=True, include_sent_mover=True,
                                               include_txt_stats=True)
                for k, v in results.items():
                    df[k] = v
                print(f"................ finish calculating scores for {len(cands)} cand ref pair(s) ................")
                if save_to:
                    df.to_csv(save_to)
                    print(
                        f"................ append results and save it to {save_to} ...............")


cand1_raw = "<t> Police say they have no objections to Sunday 's Manchester derby starting at 4 pm . </t> <t> Chief Superintendent John O'Hare says the kick - off was agreed by all parties . </t> <t> Merseyside Police launched a legal challenge after Everton v Liverpool match started at 5.30 pm . </t> <t> The last time United and City met in a late kick - off for a weekend match was in the FA Cup semi - final at Wembley in 2011 . </t> <t> 34 arrests were made amid scenes some described as ` a free for all ' . </t>"
ref1_raw = "<t> manchester united take on manchester city on sunday . </t>  <t> match will begin at 4pm local time at united 's old trafford home . </t>  <t> police have no objections to kick-off being so late in the afternoon . </t>  <t> last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . </t>"
cand1 = "Police say they have no objections to Sunday 's Manchester derby starting at 4 pm .  Chief Superintendent John O'Hare says the kick - off was agreed by all parties .  Merseyside Police launched a legal challenge after Everton v Liverpool match started at 5.30 pm .  The last time United and City met in a late kick - off for a weekend match was in the FA Cup semi - final at Wembley in 2011 .  34 arrests were made amid scenes some described as ` a free for all ' . "
ref1 = "manchester united take on manchester city on sunday .   match will begin at 4pm local time at united 's old trafford home .   police have no objections to kick-off being so late in the afternoon .   last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . "

cand1_list = ["Police say they have no objections to Sunday 's Manchester derby starting at 4 pm .",
              "Chief Superintendent John O'Hare says the kick - off was agreed by all parties .",
              "Merseyside Police launched a legal challenge after Everton v Liverpool match started at 5.30 pm .",
              "The last time United and City met in a late kick - off for a weekend match was in the FA Cup semi - final at Wembley in 2011 .",
              "34 arrests were made amid scenes some described as ` a free for all ' . "]
ref1_list = ["manchester united take on manchester city on sunday .",
             "match will begin at 4pm local time at united 's old trafford home .",
             "police have no objections to kick-off being so late in the afternoon .",
             "last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . "]

# cand1_newline = "Police say they have no objections to Sunday 's Manchester derby starting at 4 pm .\n Chief Superintendent John O'Hare says the kick - off was agreed by all parties .\n  Merseyside Police launched a legal challenge after Everton v Liverpool match started at 5.30 pm .\n  The last time United and City met in a late kick - off for a weekend match was in the FA Cup semi - final at Wembley in 2011 .\n  34 arrests were made amid scenes some described as ` a free for all ' . "
# ref1_newline = "manchester united take on manchester city on sunday .\n   match will begin at 4pm local time at united 's old trafford home .\n   police have no objections to kick-off being so late in the afternoon .\n   last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . "
# cand2 = "the boy is walking"
# ref2 = "the boy is walking."

# cand1_newline = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch . but that does n't prevent one man from dipping his hand in the fish tank and giving his blood parrot cichlid a stroke ."
# ref1_newline = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch . but that does n't prevent one man from dipping his hand in the fish tank and giving his blood parrot cichlid a stroke ."
# cand2 = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch . but that does n't prevent one man from dipping his hand in the fish tank and giving his blood parrot cichlid a stroke ."
# ref2 = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch . but that does n't prevent one man from dipping his hand in the fish tank and giving his blood parrot cichlid a stroke ."

cand1_newline = "cats and dogs have the advantage over marine pets in that they can interact with humans through the sense of touch."
ref1_newline = "cats and dogs can interact with humans through the sense of touch, therefore they have the advantage over marine pets."
# ref1_newline="by intercting with humans through the sense of touch, cats and dogs have the advantage over marine pets."
# ref1_newline="through touching, humans establish a closer relationship than marine pets."
cand2 = cand1_newline
ref2 = ref1_newline

summeval_cand1 = """
a\&e networks are remaking the series , to air in 2016 . 
the three networks will broadcast a remake of the saga of kunta kinte . 
the `` roots '' is the epic episode of the african-american slave and his descendants . 
the series of `` original '' and `` contemporary '' will be the new version of the original version ."""
summeval_cand2 = """
`` roots , '' the epic miniseries about an african-american slave and his descendants , 
had a staggering audience of over 100 million viewers back in 1977 . a\&e networks are remaking the miniseries , 
to air in 2016 . levar burton , who portrayed kinte in the original , will co-executive produce the new miniseries .
"""
summeval_cand3 = """
`` roots , '' the epic miniseries about an african-american slave and his descendants , 
had a staggering audience of over 100 million viewers back in 1977 . now a\&e networks are remaking the miniseries , 
to air in 2016 . producers will consult scholars in african and african-american history for added authenticity .
"""

summeval_cand4 = """
`` roots , '' the epic miniseries about an african-american slave and his descendants . 
a\&e , lifetime and history -lrb- formerly the history channel -rrb- announced thursday that the three networks would simulcast a remake of the saga . 
levar burton will co-executive produce the new miniseries . `` original '' producers will consult scholars in african and african-american history .
"""
summeval_ref = """
The A\&E networks are remaking the blockbuster "Roots" miniseries, to air in 2016. 
The epic 1977 miniseries about an African-American slave had 100 million viewers.
"""
summeval_cands = [summeval_cand1, summeval_cand2, summeval_cand3, summeval_cand4]
summeval_refs = [summeval_ref, summeval_ref, summeval_ref, summeval_ref]

example_cand1 = """
a&e , lifetime and history are remaking the miniseries , to air in 2016 . `` roots , '' levar burton , will co-executive produce the new miniseries . alex haley 's `` contemporary '' novel is `` original '' novel ."""
example_ref1 = """
The A&E networks are remaking the blockbuster "Roots" miniseries, to air in 2016. The epic 1977 miniseries about an African-American slave had 100 million viewers.
"""

cand1_paul = "paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scored the tottenham midfielder in the 89th minute . paul merson had another dig at andros townsend after his appearance . the midfielder had been brought on to the england squad last week . click here for all the latest arsenal news news ."
ref1_paul = "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up."


def example_rouge(scorer: MetricScorer, candidate="", reference=""):
    if scorer:
        # rouge_dict = scorer.get_rouge(cand1, ref1)
        rouge_dict = scorer.get_rouge(summeval_cands, summeval_refs, batch_mode=True)
        # rouge_dict = scorer.get_rouge(cand1_list, ref1_list,batch_mode=True)
        print("rouge_dict: ")
        print(rouge_dict)


def example_js_eval(scorer: MetricScorer, candidate="", reference=""):
    if scorer:
        cand_cnn1 = "paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scored the tottenham midfielder in the 89th minute . paul merson had another dig at andros townsend after his appearance . the midfielder had been brought on to the england squad last week . click here for all the latest arsenal news news ."
        cand_cnn2 = "paul merson has restarted his row with andros townsend . the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scores england 's equaliser in their 1-1 friendly draw with italy in turin ."
        cand_cnn3 = "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley on sunday . townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley . townsend hit back at merson on twitter after scoring for england against italy ."
        ref_cnn1 = "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up."
        # js_dict = scorer.get_JS([cand_cnn1, cand_cnn2, cand_cnn3], [ref_cnn1, ref_cnn1, ref_cnn1], batch_mode=True)
        example_cand1 = cand1_paul
        example_ref1 = ref1_paul
        js_dict = scorer.get_JS([example_cand1, example_cand1], [example_ref1, example_ref1], batch_mode=True)
        print(f"js_dict: {js_dict}")


def example_rouge_we(scorer: MetricScorer, candidate="", reference=""):
    if scorer:
        example_cand1 = """
        Shocking footage: a driver was found awake in his car. He says in the video: dead set can't not wake him up...off his head. The footage was went to Ray Hadley of 2gb ratio and uploaded on youtube on monday. 2gb says the driver woke upon the m1 motoway north of sydney.
        """
        example_ref1 = """
        An unimpressed motorist documented how he tried to wakeup the driver. He says in the video : dead set can't wake him up ... off his head'. The footage was sent to ray hadley of 2 gb radio and uploaded on youtube on monday. 2 gb says the driver had passed out on the m1 motorway north of sydney.
        """
        example_cand1 = cand1_paul
        example_ref1 = ref1_paul
        score1 = scorer.get_rouge_we(example_cand1, example_ref1, n_gram=1)
        score2 = scorer.get_rouge_we(example_cand1, example_ref1, n_gram=2)
        score3 = scorer.get_rouge_we(example_cand1, example_ref1, n_gram=3)
        print("rouge_we n_gram=1: ")
        print(score1)
        print("rouge_we n_gram=2: ")
        print(score2)
        print("rouge_we n_gram=3: ")
        print(score3)


def example_bertscore(scorer: MetricScorer, candidate="", reference=""):
    # score_dict = scorer.get_bert_score(cand1, ref1)
    # score_dict = scorer.get_bert_score([example_cand1, example_cand1], [example_ref1, example_ref1], batch_mode=True,
    #                                    aggregate=False)
    example_cand1 = cand1_paul
    example_ref1 = ref1_paul
    score_dict = scorer.get_bert_score([example_cand1, example_cand1], [example_ref1, example_ref1], batch_mode=True,
                                       aggregate=False)
    print("bert_score dict: ")
    print(score_dict)
    # print(scorer.get_bert_score(CANDS, REFS,batch_mode=True))
    # avgP = sum([0.9843302369117737, 0.9832239747047424, 0.9120386242866516]) / 3
    # avgR = sum([0.9823839068412781, 0.9732863903045654, 0.920428991317749]) / 3
    # avgF = sum([0.9833561182022095, 0.9782299995422363, 0.916214644908905]) / 3
    # print(avgP, avgR, avgF)


def example_chrfpp(scorer: MetricScorer, candidate="", reference=""):
    score = scorer.get_chrfpp(cand1_newline, ref1_newline)
    print('single example')
    print(score)
    # print("cand2 ref2: ")
    # score = scorer.get_chrfpp(cand2, ref2)
    # print(score)

    # score = scorer.get_chrfpp(cand1_newline, ref1_newline, batch_mode=True)
    # print("batch")
    # print(score)
    # def test_score(self):
    #     metric = ChrfppMetric()
    #     score = metric.evaluate_batch(CANDS_chrfpp, REFS_chrfpp)
    #     ref = {'chrf': 0.38735906038936213}
    #     self.assertTrue((score['chrf'] - ref['chrf']) < EPS)
    #     single_score = metric.evaluate_example(CANDS_chrfpp[0], REFS_chrfpp[0])
    #     ref_single = {'chrf': 0.6906099983606945}
    #     self.assertTrue((single_score['chrf'] - ref_single['chrf']) < EPS)


def example_cider(scorer: MetricScorer, candidate="", reference=""):
    # score = scorer.get_cider(cand1_newline, ref1_newline, batch_mode=True)
    print("cand2 ref2: ")
    score = scorer.get_cider(cand2, ref2, tokenize=True)

    print(score)


def example_moverscore(scorer: MetricScorer, candidate="", reference=""):
    example_cand1 = cand1_paul
    example_ref1 = ref1_paul
    score = scorer.get_moverscore([example_cand1], [example_ref1], batch_mode=True)
    print("version 1, default setting: ", score)
    score = scorer.get_moverscore(use_default=False, version=2, cands=example_cand1, refs=example_ref1)
    print("version 2: ", score)
    # score = scorer.get_moverscore(cand2, cand2)
    # print("cand2 ref2: ")
    # print(score)


def example_sent_mover(scorer: MetricScorer, candidate="", reference=""):
    # score = scorer.get_sent_mover(cand1_newline, ref1_newline, metric_type="sms")
    # print('example: sms')
    # print(score)
    #
    # score = scorer.get_sent_mover(cand1_newline, ref1_newline, metric_type="wms")
    # print('example: wms')
    # print(score)
    #
    # score = scorer.get_sent_mover(cand1_newline, ref1_newline, metric_type="s+wms")
    # print('example: s+wms')
    # print(score)
    cand2 = cand1_paul
    ref2 = ref1_paul
    score = scorer.get_sent_mover(cand2, ref2, metric_type="sms")
    print('example: sms')
    print(score)

    score = scorer.get_sent_mover(cand2, ref2, metric_type="wms")
    print('example: wms')
    print(score)

    score = scorer.get_sent_mover(cand2, ref2, metric_type="s+wms")
    print('example: s+wms')
    print(score)


def example_meteor(scorer: MetricScorer, candidate="", reference=""):
    # score = scorer.get_meteor(cand1_newline, ref1_newline)
    example_cand1 = cand1_paul
    example_ref1 = ref1_paul
    score = scorer.get_meteor(example_cand1, example_ref1)
    print(score)


def example_bleu(scorer: MetricScorer, candidate="", reference=""):
    cand1_newline = cand1_paul
    ref1_newline = ref1_paul
    score = scorer.get_bleu(cand1_newline, ref1_newline)
    print(score)
    # score = scorer.get_bleu(cand1_newline, cand1_newline)
    # print(score)


def example_stats_temp(scorer: MetricScorer):
    # score = scorer.get_txt_stats("hello world this is fine", 'hi there how are')
    nlp = spacy.load('en_core_web_sm')
    # nlp = spacy.load('en_core_web_md')
    disable = ["tagger", "textcat"]
    cand_summaries_spacy = [nlp(example_cand1, disable=disable)]
    ref_summaries_spacy = [nlp(example_ref1, disable=disable)]
    # cand_spacy_stats = [tok.text for tok in cand_summaries_spacy]
    # ref_spacy_stats = [tok.text for tok in ref_summaries_spacy]
    cand_spacy_stats = " ".join(tok.text for tok in cand_summaries_spacy)
    ref_spacy_stats = " ".join(tok.text for tok in ref_summaries_spacy)
    score = scorer.get_txt_stats(cand_spacy_stats, ref_spacy_stats)
    print(score)


def example_stats(scorer: MetricScorer):
    # score = scorer.get_txt_stats("hello world this is fine", 'hi there how are')
    # nlp = spacy.load('en_core_web_sm')
    # # nlp = spacy.load('en_core_web_md')
    # disable = ["tagger", "textcat"]
    # stats = metric.evaluate_example(CANDS[0], ARTICLE)
    #
    # cand_summaries_spacy = [nlp(example_cand1, disable=disable)]
    # ref_summaries_spacy = [nlp(example_ref1, disable=disable)]
    # # cand_spacy_stats = [tok.text for tok in cand_summaries_spacy]
    # # ref_spacy_stats = [tok.text for tok in ref_summaries_spacy]
    # cand_spacy_stats = " ".join(tok.text for tok in cand_summaries_spacy)
    # ref_spacy_stats = " ".join(tok.text for tok in ref_summaries_spacy)
    from summ_eval.test_util import CANDS, ARTICLE
    score = scorer.get_txt_stats(cand1_paul, ref1_paul)
    print(score)


def example_s3(scorer: MetricScorer):
    score = scorer.get_s3([cand1_newline], [ref1_newline], batch_mode=True)
    print(score)


def example_sent(scorer: MetricScorer):
    example_rouge(scorer)
    example_rouge_we(scorer)
    example_bertscore(scorer=scorer)
    example_chrfpp(scorer)
    example_cider(scorer=scorer)
    example_moverscore(scorer=scorer)
    example_sent_mover(scorer=scorer)
    example_meteor(scorer=scorer)
    example_bleu(scorer=scorer)
    example_stats(scorer=scorer)


# CANDS = [
#     "28-year-old chef found dead in San Francisco mall",
#     "A 28-year-old chef who recently moved to San Francisco was found dead in the staircase of a local shopping center.",
#     "The victim's brother said he cannot imagine anyone who would want to harm him,\"Finally, it went uphill again at him.\"",
# ]
# REFS = [
#     "28-Year-Old Chef Found Dead at San Francisco Mall",
#     "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.",
#     "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally going well for him.\""
# ]

def cal_scores_df_all(scorer: MetricScorer):
    src_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/examples/example_with_scores/abs_presumm_ext_abs.csv"
    # results = scorer.cal_scores_from_csv(src_path=src_path, save_to=src_path)
    src2 = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/examples/example_with_scores/ext_bart_out_ext.csv"
    summeval_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix/summeval_mix_all_metrics.csv"
    save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix/newJS_summeval_mix_all_metrics.csv"
    # scorer.cal_scores_from_csv(src_path=summeval_path, save_to=save_to, clean_text=False, cand_col="decoded",
    #                            ref_col="reference")
    scorer.cal_JS_scores_from_csv(src_path=summeval_path, save_to=save_to, clean_text=False, cand_col="decoded",
                                  ref_col="reference")
    print(f"************ Done calucation *****************")


def cal_scores_for_realsumm(scorer: MetricScorer):
    src_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/abs_models/abs_amr_scores_splitted/bart_out.csv"
    save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/abs_models/abs_amr_s3/bart_out_extra.csv"

    # src_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/abs_models/abs_amr_s3/bart_out_original.csv"
    # save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/abs_models/abs_amr_s3/bart_out_more_scores.csv"

    scorer.cal_scores_from_csv(src_path=src_path, save_to=save_to)


def cal_scores_realsumm_from_dir(scorer: MetricScorer):
    dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/amr_splitted_measure/add_wms"
    files = get_files_in_dir(dir, sort_by_name=True)
    for file in files:
        if not file.endswith('.csv'): continue
        fp = os.path.join(dir, file)
        # df = load_df(fp)
        scorer.cal_scores_from_csv_for_realsumm(src_path=fp, save_to=fp)


def batch_cal_scores_realsumm_from_txt(scorer: MetricScorer):
    cand_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/cand_ref_txt/cand_realsumm_abs_all_metrics.txt"
    ref_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/cand_ref_txt/ref_realsumm_abs_all_metrics.txt"
    save_to = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/realsumm/recalculated_realsumm/realsumm_abs.csv"
    cands = []
    refs = []
    with open(cand_path, 'r') as cand_f, open(ref_path, 'r') as ref_f:
        for cand, ref in zip(cand_f, ref_f):
            cands.append(cand)
            refs.append(ref)

    scorer.cal_batch_scores_from_txt(cands[:3], refs[:3], save_to=save_to)


def cal_scores_for_df(scorer: MetricScorer):
    src_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/external/experiments/all_data/summeval/models_with_docid/with_s3/abs_ext_mix/summeval_mix_all_metrics.csv"
    save_to = "/Users/jackz/Desktop/data/scorer_cal.csv"
    print(f"start cal")
    scorer.cal_scores_from_csv(src_path=src_path, save_to=save_to, cand_col="decoded", ref_col="reference")


if __name__ == '__main__':
    set_path()
    # EPS = 1e-5
    scorer = MetricScorer()
    # example_rouge(scorer)
    # example_rouge_we(scorer)
    # example_bertscore(scorer=scorer)
    # example_chrfpp(scorer)
    # example_cider(scorer=scorer)
    # example_moverscore(scorer=scorer)
    # example_sent_mover(scorer=scorer)
    # example_meteor(scorer=scorer)
    # example_bleu(scorer=scorer)
    # example_sent(scorer)
    # cal_scores_df(scorer)
    # cal_scores_for_realsumm(scorer)
    # example_chrfpp(scorer)
    # example_cider(scorer)
    # example_stats(scorer)
    # example_s3(scorer=scorer)
    # cal_scores_for_realsumm(scorer)
    # cal_scores_realsumm_from_dir(scorer)
    # batch_cal_scores_realsumm_from_txt(scorer=scorer)
    # example_js_eval(scorer=scorer)
    # example_rouge_we(scorer=scorer)
    # example_js_eval(scorer=scorer)
    # example_bertscore(scorer=scorer)
    # example_moverscore(scorer=scorer)
    # # example_sent_mover(scorer=scorer)
    # example_meteor(scorer=scorer)
    # example_bleu(scorer=scorer)
    # example_stats(scorer=scorer)
    cal_scores_for_df(scorer=scorer)