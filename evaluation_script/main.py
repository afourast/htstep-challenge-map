import random
import json 
import numpy as np


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    if phase_codename == 'test':
        evaluate_recall(test_annotation_file, user_submission_file, phase_codename, **kwargs)

    elif phase_codename == 'map_test_unseen':
        evaluate_map(test_annotation_file, user_submission_file, phase_codename, **kwargs)

        
def evaluate_map(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    split = 'test_seen'
    tiou_thresholds = np.linspace(0.3, 0.7, 5)

    from eval_detection import evaluate_article_grounding
    evaluate_article_grounding(test_annotation_file, user_submission_file, split, tiou_thresholds)


def evaluate_recall(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}

    test_annots = json.load(open(test_annotation_file))
    submission = json.load(open(user_submission_file))

    if not len(submission) == len(test_annots):
        print(f'Missing some annotations -- expected {len(test_annots)}, but received {len(submission)}')

    recalls = []

    print(f'Evaluating Recall@1 on {len(test_annots)} samples.')

    for vid in test_annots:

        if vid not in submission:
            print(f'{vid} not found in annotations')
            continue

        n_txt = len(submission[vid])
        gt = test_annots[vid]
        assert len(gt['segments']) == n_txt

        for text_idx in range(n_txt):

            if not gt['aligned'][text_idx]:
                continue

            pred_timestamp = submission[vid][text_idx]

            segments_aligned = gt['segments'][text_idx]
            retrieved = False
            for segment in segments_aligned:

                if segment[1] - segment[0] < 1: # ignore segments smaller than 1 second 
                    continue

                if segment[0] <= pred_timestamp <= segment[1]:
                    retrieved = True
                    break
            recalls.append(retrieved)

    recall = np.mean(recalls)

    output["result"] = [
        {
            "test_split": {
                "Recall@1": recall,
            }
        }
    ]

    output["submission_result"] = output['result'][0]["test_split"]
    # output["submission_result"] = output["result"][0]


    print(output)

    return output
