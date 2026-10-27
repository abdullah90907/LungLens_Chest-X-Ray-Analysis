import numpy as np
from .schemas import EvaluationRequest


def evaluate_observations(payload: EvaluationRequest, references):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    grouped = {label:[] for label in references}
    seen, cases = set(), set()
    for observation in payload.observations:
        if observation.label not in grouped:
            raise ValueError('Evaluation contains an unsupported finding label')
        key = (observation.case_id,observation.label)
        if key in seen:
            raise ValueError('Duplicate case/finding observations are not allowed')
        seen.add(key)
        cases.add(observation.case_id)
        grouped[observation.label].append((observation.truth, observation.score))
    metrics = {}
    for label, observations in grouped.items():
        if not observations:
            continue
        truth, scores = np.asarray(observations,dtype=float).T
        cutoff = float(references[label])
        predicted = scores >= cutoff
        tp = int(np.sum(predicted & (truth == 1)))
        tn = int(np.sum(~predicted & (truth == 0)))
        fp = int(np.sum(predicted & (truth == 0)))
        fn = int(np.sum(~predicted & (truth == 1)))
        sensitivity = tp/(tp+fn) if tp+fn else None
        specificity = tn/(tn+fp) if tn+fp else None
        both = len(np.unique(truth)) == 2
        roc, pr = [], []
        if both:
            fpr,tpr,_ = roc_curve(truth,scores)
            indices = np.unique(np.linspace(0,len(fpr)-1,min(101,len(fpr))).astype(int))
            roc = [{'false_positive_rate':float(fpr[i]),'true_positive_rate':float(tpr[i])} for i in indices]
            precision,recall,_ = precision_recall_curve(truth,scores)
            indices = np.unique(np.linspace(0,len(recall)-1,min(101,len(recall))).astype(int))
            pr = [{'recall':float(recall[i]),'precision':float(precision[i])} for i in indices]
        bins = []
        for i in range(10):
            selected = (scores >= i/10) & ((scores < (i+1)/10) if i < 9 else (scores <= 1))
            if selected.any():
                bins.append({'score_lower':i/10,'score_upper':(i+1)/10,'count':int(selected.sum()),
                             'mean_score':float(scores[selected].mean()),'observed_positive_fraction':float(truth[selected].mean())})
        warnings = []
        if not both:
            warnings.append('Both truth classes are needed for AUROC and meaningful precision-recall comparison')
        if len(truth) < 20:
            warnings.append('Small sample: metrics are descriptive and may be unstable')
        metrics[label] = {'n':len(truth),'positives':int(truth.sum()),'reference_threshold':cutoff,
            'auroc':float(roc_auc_score(truth,scores)) if both else None,
            'average_precision':float(average_precision_score(truth,scores)) if both else None,
            'sensitivity':sensitivity,'specificity':specificity,
            'precision':tp/(tp+fp) if tp+fp else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'balanced_accuracy':(sensitivity+specificity)/2 if both else None,
            'brier_score':float(np.mean((scores-truth)**2)),
            'tp':tp,'tn':tn,'fp':fp,'fn':fn,'roc_curve':roc,'precision_recall_curve':pr,
            'score_bins':bins,'warnings':warnings}
    return {'schema_version':'1.0','cases':len(cases),'observations':len(seen),
            'model_weights':payload.model_weights,'checkpoint_sha256':payload.checkpoint_sha256,
            'provenance_verified':False,'scores_are_calibrated':False,'metrics':metrics,
            'curve_max_points':101,
            'notice':'Aggregate research metrics from supplied labels and scores. Provenance, patient independence, annotation quality and training-set overlap are not verified. Score bins and Brier scores are diagnostics, not evidence of calibrated probabilities. No case identifiers are returned.'}
