import os
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

cand1_newline = "Police say they have no objections to Sunday 's Manchester derby starting at 4 pm .\n Chief Superintendent John O'Hare says the kick - off was agreed by all parties .\n  Merseyside Police launched a legal challenge after Everton v Liverpool match started at 5.30 pm .\n  The last time United and City met in a late kick - off for a weekend match was in the FA Cup semi - final at Wembley in 2011 .\n  34 arrests were made amid scenes some described as ` a free for all ' . "
ref1_newline = "manchester united take on manchester city on sunday .\n   match will begin at 4pm local time at united 's old trafford home .\n   police have no objections to kick-off being so late in the afternoon .\n   last late afternoon weekend kick-off in the manchester derby saw 34 fans arrested at wembley in 2011 fa cup semi-final . "
cand2 = "the boy is walking"
ref2 = "the boy is walking."


# cand1="the boy is walking"
# ref1=cand1

def example_rouge(scorer: MetricScorer):
    if scorer:
        # rouge_dict = scorer.get_rouge(cand1, ref1)
        rouge_dict = scorer.get_rouge(cand1_newline, ref1_newline)
        # rouge_dict = scorer.get_rouge(cand1_list, ref1_list,batch_mode=True)
        print("rouge_dict: ")
        print(rouge_dict)


def example_rouge_we(scorer: MetricScorer):
    if scorer:
        score1 = scorer.get_rouge_we(cand1_newline, ref1_newline, n_gram=1)
        score2 = scorer.get_rouge_we(cand1_newline, ref1_newline, n_gram=2)
        print("rouge_we n_gram=1: ")
        print(score1)
        print("rouge_we n_gram=2: ")
        print(score2)


def example_bertscore(scorer: MetricScorer):
    # score_dict = scorer.get_bert_score(cand1, ref1)
    score_dict = scorer.get_bert_score(cand1_newline, ref1_newline)

    print("bert_score dict: ")
    print(score_dict)
    # print(scorer.get_bert_score(CANDS, REFS,batch_mode=True))
    # avgP = sum([0.9843302369117737, 0.9832239747047424, 0.9120386242866516]) / 3
    # avgR = sum([0.9823839068412781, 0.9732863903045654, 0.920428991317749]) / 3
    # avgF = sum([0.9833561182022095, 0.9782299995422363, 0.916214644908905]) / 3
    # print(avgP, avgR, avgF)


def example_chrfpp(scorer: MetricScorer):
    score = scorer.get_chrfpp(cand1_newline, ref1_newline)
    print('single example')
    print(score)
    print("cand2 ref2: ")
    score = scorer.get_chrfpp(cand2, ref2)
    print(score)

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


def example_cider(scorer: MetricScorer):
    # score = scorer.get_cider(cand1_newline, ref1_newline, batch_mode=True)
    print("cand2 ref2: ")
    score = scorer.get_cider(cand2, ref2, tokenize=True)

    print(score)


def example_moverscore(scorer: MetricScorer):
    score = scorer.get_moverscore(cand1_newline, ref1_newline)
    print("version 1, default setting: ", score)
    score = scorer.get_moverscore(use_default=False, version=2, cands=cand1_newline, refs=ref1_newline)
    print("version 2: ", score)
    # score = scorer.get_moverscore(cand2, cand2)
    # print("cand2 ref2: ")
    # print(score)


def example_sent_mover(scorer: MetricScorer):
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

    score = scorer.get_sent_mover(cand2, cand2, metric_type="sms")
    print('example: sms')
    print(score)

    score = scorer.get_sent_mover(cand2, cand2, metric_type="wms")
    print('example: wms')
    print(score)

    score = scorer.get_sent_mover(cand2, cand2, metric_type="s+wms")
    print('example: s+wms')
    print(score)


def example_meteor(scorer: MetricScorer):
    # score = scorer.get_meteor(cand1_newline, ref1_newline)
    score = scorer.get_meteor(cand1_newline, cand1_newline)
    print(score)


def example_bleu(scorer: MetricScorer):
    score = scorer.get_bleu(cand1_newline, ref1_newline)
    print(score)
    score = scorer.get_bleu(cand1_newline, cand1_newline)
    print(score)


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
    example_bleu(scorer=scorer)
