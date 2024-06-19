import sklearn
import numpy as np
import matplotlib.pyplot as plt

# Args:
# y_true (ndarray): Ground truth labels (0 or 1).
# y_pred (ndarray): Predicted labels (0 or 1).

def coherence_check(true, pred):
    new_pred = pred[:]
    if len(pred) > len(true):
        new_pred = pred[:len(true)]
    if len(pred) < len(true):
        pad_length = len(true) - len(pred)
        new_pred = np.array(pred.tolist() + [0] * pad_length)
    return new_pred

# F-1 score
# Harmonic mean of the precision and recall
def f1_score(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred)
# F-beta score
def fbeta_score(y_true, y_pred, beta=1):
    return sklearn.metrics.f_beta_score(y_true, y_pred, beta=beta)

# Segment-based metrics
def segment_metrics(y_true, y_pred, num_segment):
    if len(y_true) != len(y_pred):
        raise ValueError("The ground truth and prediction does not have the same length")
    # Initialize variables (S, I, D N does not apply in binary classification)
    tp, fp, fn = 0, 0, 0
    # Divide into segments
    true_segments = np.array_split(y_true, num_segment)
    pred_segments = np.array_split(y_pred, num_segment)
    # Calculation of tp, fp, fn
    for true_segment, pred_segment in zip(true_segments, pred_segments):
        threshold = len(true_segment) / 2
        # Get the label in the segment
        true_label = 1 if sum(true_segment) > threshold else 0
        pred_label = 1 if sum(pred_segment) > threshold else 0
        if true_label == pred_label and pred_label == 1:
            tp += 1
        if true_label != pred_label and pred_label == 1:
            fp += 1
        if true_label != pred_label and pred_label == 0:
            fn += 1
    # Calculation of F-Score
    P = tp / (tp + fp) if (tp + fp) != 0 else 0
    R = tp / (tp + fn) if (tp + fn) != 0 else 0
    F = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return {'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'f_score': F}    

# Event-based metrics
def event_metrics(y_true, y_pred, tolerance, overlap_threshold=0.7):
    # Create empty list for storing true events
    true_events = []
    # Initilize start index
    start = None
    for i, label in enumerate(y_true):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            true_events.append((start, i - 1))
            start = None
    
    if start is not None:
        true_events.append((start, len(y_true) - 1))

    pred_events = []
    start = None
    for i, label in enumerate(y_pred):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            pred_events.append((start, i - 1))
            start = None

    if start is not None:
        pred_events.append((start, len(pred_events) - 1))
        

    # Highlight overlapping events
    # Intialize true positive and overlap events
    tp, fp, fn = 0, 0, 0
    counted_events = []
    fake_events = []
    undetected_events = []
    pred_check = pred_events[:]

    for true_event in true_events:
        tp_event = 0
        for pred_event in pred_events:
            lower_bound = true_event[0] - tolerance
            upper_bound = true_event[1] + tolerance
            # Calculate overlap rate
            overlap_rate = 0
            if lower_bound <= pred_event[0] and upper_bound >= pred_event[1]:
                overlap_start = max(true_event[0], pred_event[0])
                overlap_end = min(true_event[1], pred_event[1])
                overlap_length = overlap_end - overlap_start + 1
                true_length = true_event[1] - true_event[0] + 1
                pred_length = pred_event[1] - pred_event[0] + 1
                overlap_rate = overlap_length / min(true_length, pred_length)
            # Range check
            if overlap_rate >= overlap_threshold:
                # True positive: correctly detected events
                pred_check.remove(pred_event)
                if tp_event == 0:
                    tp_event = 1
                    counted_events.append((true_event[0], true_event[1]))

        # False negative: events in true label that have not been correctly detected according to the definition
        if tp_event == 0:
            fn += 1
            undetected_events.append((true_event[0], true_event[1]))

        tp += tp_event
    # False positive: events in prediction that are not correct according to the definition
    if pred_check:
        for pred_event in pred_check: 
            fp += 1
            fake_events.append((pred_event[0], pred_event[1]))

    if tp == 0 and fn == 0 and fp == 0:
        F = 1
    else:
        # Calculation of F-Score
        P = tp / (tp + fp) if (tp + fp) != 0 else 0
        R = tp / (tp + fn) if (tp + fn) != 0 else 0
        F = 2 * P * R / (P + R) if (P + R) != 0 else 0
    return {'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'f_score': F,
            'counted_events': counted_events,
            'fake_events': fake_events,
            'undetected_events': undetected_events}

def event_visualization(y_true, y_pred, counted_events, fake_events, undetected_events):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_true)), y_true, label='True Label')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Label')

    for event in counted_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='green', label='Overlap event')
    for event in fake_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='red', label='Fake event')
    for event in undetected_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='blue', label='Undetected event')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Overlapping Events Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Reference: https://doi.org/10.3390/app6060162