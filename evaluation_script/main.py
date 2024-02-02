import random
import json 
import numpy as np


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    evaluate_map(test_annotation_file, user_submission_file, phase_codename, **kwargs)

        
def evaluate_map(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    assert phase_codename in ['test_seen', 'test_unseen', 'val_unseen']

    split = phase_codename

    tiou_thresholds = np.linspace(0.3, 0.7, 5)

    from eval_detection import evaluate_article_grounding
    grounding_map = evaluate_article_grounding(test_annotation_file, user_submission_file, split, tiou_thresholds)

    output = {}

    output["result"] = [
        {
            f"{split}_split": {
                "mAP": grounding_map,
            }
        }
    ]

    # output["submission_result"] = output['result'][0]["test_split"]
    output["submission_result"] = output['result'][0][f"{split}_split"]
    # output["submission_result"] = output["result"][0]

    print(output)

    return output


