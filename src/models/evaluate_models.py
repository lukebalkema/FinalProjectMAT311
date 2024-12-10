from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
    
    cm = confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    specificity = TN / (TN + FP)
    
    report = classification_report(y_val, y_pred)
    
    metrics = {
        'cm': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'report': report
    }
    
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_auc'] = roc_auc
    
    return metrics
