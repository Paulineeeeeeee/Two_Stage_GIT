# -*- coding=utf-8 -*-
# author: w61
# Test for several ways to compute the score of the generated words.
# from coco_caption import pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import evaluate
# from pycocoevalcap.spice.spice import Spice
# from pycocoevalcap.wmd.wmd import WMD

class Scorers():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
    

    def transform(self):
        # Initialize empty dictionaries to hold the transformed reference and prediction data
        ref_dict = {}
        pred_dict = {}

        for i, sublist in enumerate(self.gt):
            key = str(i+1)
            ref_dict[key] = sublist

        for i, sublist in enumerate(self.ref):
            key = str(i+1)
            pred_dict[key] = [sublist]
        
        # Overwrite the original gt and ref data with the transformed dictionaries
        self.gt = ref_dict
        self.ref = pred_dict


    def compute_scores(self):
        self.transform()
        total_scores = {}
            # 檢查字典中是否存在有效的 caption
        def has_valid_caption(captions_dict):
            for captions in captions_dict.values():
                if any(caption.strip() for caption in captions):
                    return True
            return False

        # 如果 ground truth 或 reference 中沒有有效 caption，返回最低分數和空的項目數
        if not has_valid_caption(self.gt) or not has_valid_caption(self.ref):
            print("有效的 caption 為空，返回所有評分指標的最低分數")
            total_scores["bleu"] = [0, 0, 0, 0]     # BLEU 指標返回 4 個分數
            total_scores["METEOR"] = 0             # METEOR 為 0
            total_scores["ROUGE_L"] = 0            # ROUGE_L 為 0
            total_scores["CIDEr"] = 0       # CIDEr 返回 (整體分數, 分數列表)
            return total_scores

        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                total_scores["bleu"] = score
            else:
                total_scores[method] = score
        return total_scores
        # for scorer, method in self.scorers:
        #     # print('computing %s score...'%(scorer.method()))
        #     score, scores = scorer.compute_score(self.gt, self.ref)
        #     if type(method) == list:
        #         # for sc, scs, m in zip(score, scores, method):
        #         #     print("%s: %0.3f"%(m, sc))
        #         total_scores["bleu"] = score #record bleu score
        #     else:
        #         # print("%s: %0.3f"%(method, score))
        #         total_scores[method] = score #record other 3 scores
        
        # print('*****DONE*****')
        # for key,value in total_scores.items():
        #     print('{}:{}'.format(key,value))

        return total_scores

def transpose(arr):
    return [list(i) for i in zip(*arr)]

def calculate_score(predictions, references, metric = 'bleu'):
    score = evaluate.load(metric)
    results = score.compute(predictions=predictions, references=references)
    # print(f"evaluate_{metric}: ",results)
    if(metric == 'rouge'):
        return results['rougeL']
    return results[metric]

if __name__ == '__main__':
    # install trace 16
    # google app trace 56
    # general trace_15
    # webshopping trace_4
    ref = ['Open app "HBO Max: Stream TV & Movies" (install if not already installed) and go to login screen',
        'Turn off notifications in google photos',
        'Open the contacts',
        'Search for the new nike air max shoes on Nike.']
    # gt = [
    #     ['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
    #     ['It is a cat.','There is a cat over there.','cat over there.']
    # ]
    gt = [
        ['The user’s goal is to download, install, and sign in to HBO Max.'],
        ['Block all Google Photos notifications.'],
        ['The user’s goal is to view and manage contact information.'],
        ["The user's goal is to search for and browse Nike products online."]
    ]
    
    scorer = Scorers(ref,gt)
    # scorer = Scorers(gt,ref)
    total_score = scorer.compute_scores()
    # total_score = cocoeval.compute_scores()
    print(f" bleu = {total_score['bleu'][3]:.5f}, CIDEr = {total_score['CIDEr']:.5f}, RougeL = {total_score['ROUGE_L']:.5f}")