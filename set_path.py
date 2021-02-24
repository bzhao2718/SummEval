import os

# export ROUGE_HOME=/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/evaluation/summ_eval/ROUGE-1.5.5/
# export PYTHONPATH=$PYTHONPATH:/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/evaluation/summ_eval/
# export CORENLP_HOME=/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/evaluation/summ_eval/stanford-corenlp-full-2018-10-05/




def set_path():
    os.environ['ROUGE_HOME'] = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/SummEval/evaluation/summ_eval/ROUGE-1.5.5/"

if __name__ == '__main__':
    set_path()