import numpy as np
import matplotlib.pyplot as plt

# Args:
# y_true (ndarray): Ground truth labels (0 or 1).
# y_pred (ndarray): Predicted labels (0 or 1).

# True positive
def true_positive(y_true, y_pred):
    tp = 0
    for label, pred in zip(y_true, y_pred):
        if label == pred and pred == 1:
            tp += 1
    return tp

# False positive
def false_positive(y_true, y_pred):
    fp = 0
    for label, pred in zip(y_true, y_pred):
        if label != pred and pred == 1:
            fp += 1
    return fp

# False negative
def false_negative(y_true, y_pred):
    fn = 0
    for label, pred in zip(y_true, y_pred):
        if label != pred and pred == 0:
            fn += 1
    return fn

# True negative
def true_negative(y_true, y_pred):
    tn = 0
    for label, pred in zip(y_true, y_pred):
        if label == pred and pred == 0:
            tn += 1
    return tn

# 1. Confusion matrix
def confusion_matrix(y_true, y_pred):
    if len(np.unique(y_true)) != 2 or len(np.unique(y_pred)) != 2:
        raise ValueError("y_true and y_pred must be binary (0 or 1) for confusion matrix calculation.")

    # Calculate confusion matrix elements
    TN = true_negative(y_true, y_pred)
    FP = false_positive(y_true, y_pred)
    FN = false_negative(y_true, y_pred)
    TP = true_positive(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

# True Positive Rate (TPR) or Recall
def true_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1] 
    FN = cm[1, 0]
    P = TP + FN 
    return TP / P if P > 0 else 0 

# True Negative Rate (TNR) or Specificity
def true_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]
    N = TN + FP  # Total Negatives
    return TN / N if N > 0 else 0

# False Positive Rate (FPR) or Fall-out
def false_positive_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0, 1]
    TN = cm[0, 0]
    N = TN + FP 
    return FP / N if N > 0 else 0

# False Negative Rate (FNR) or Miss Rate
def false_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    FN = cm[1, 0] 
    TP = cm[1, 1]
    P = TP + FN 
    return FN / P if P > 0 else 0

# Precision
def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0

# 2. Accuracy
def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]
    return (TP + TN) / (TP + FN + FP + TN)

# Balanced accuracy
def balanced_accuracy(y_true, y_pred, weight=0.5):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]
    return weight * TP / (TP + FN) + (1 - weight) * TN / (TN + FP)

# Error rate
def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)

# 3. F-1 score
# Harmonic mean of the precision and recall
def f_1_score(y_true, y_pred):
    return 2 * (precision(y_true, y_pred) * true_positive_rate(y_true, y_pred)) / (precision(y_true, y_pred) + true_positive_rate(y_true, y_pred))
# F-beta score
def f_beta_score(y_true, y_pred, beta=1):
    return (1 + beta^2) * (precision(y_true, y_pred) * true_positive_rate(y_true, y_pred)) / (precision(y_true, y_pred) * beta^2 + true_positive_rate(y_true, y_pred))

# 4. Receiver Operating Characteristic Curve (ROC)
# 5. Area Under Curve (AUC)
# 6. Precision-Recall Curve

# 7. Segment-based metrics
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
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    return {'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'f_score': F}    

# 8. Event-based metrics
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
    
    # Calculation of F-Score
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    
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

