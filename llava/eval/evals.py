import nltk
from rouge import Rouge
from bert_score import score
from alignscore import AlignScore


def calculate_scores(text, reference_text):
    '''
    Calculate the BLEU, ROUGE and BERTScore

    # Example usage
    text = "your text here"
    reference_text = "your reference text here"
    print(calculate_scores(text, reference_text))

    '''
    # BLEU score
    hypothesis = nltk.word_tokenize(text)
    reference = nltk.word_tokenize(reference_text)
    BLEU_score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    # ROUGE score
    rouge = Rouge()
    ROUGE_score = rouge.get_scores(text, reference_text, avg=True)

    # BERTScore
    P, R, F1 = score([text], [reference_text], lang="en", verbose=True)
    BERTScore = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

    return {"BLEU": BLEU_score, "ROUGE": ROUGE_score, "BERTScore": BERTScore}
    # return {"BLEU": BLEU_score, "BERTScore": BERTScore}


def align_score(text, reference_text, scorer=None):
    '''
    from alignscore import AlignScore

    scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='/path/to/checkpoint', evaluation_mode='nli_sp')
    score = scorer.score(contexts=['hello world.'], claims=['hello world.'])
    '''
    if scorer is None:
        scorer = AlignScore(model='roberta-large', 
                            batch_size=1, device='cuda:1', 
                            ckpt_path='/shared/nas/data/m1/jk100/code/AlignScore/ckpt/AlignScore-large.ckpt',
                            evaluation_mode='nli')
    score = scorer.score(contexts=[reference_text], claims=[text])
    print(f"[ Score: {score} ]")
    return score[0]