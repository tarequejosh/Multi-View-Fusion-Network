# PMRAM B2 External Classification Report

Evaluated Images: 1410
Accuracy: 0.9206
Macro F1: 0.9204
Macro AUC: 0.9730
ECE: 0.0370
Brier: 0.1364

## Per-Class Metrics
- glioma: P=0.8779, R=0.9135, F1=0.8954
- meningioma: P=0.9563, R=0.9358, F1=0.9459
- pituitary: P=0.8857, R=1.0000, F1=0.9394
- no_tumor: P=0.9895, R=0.8270, F1=0.9010

## Top 5 High-Confidence Errors
- True: no_tumor, Pred: glioma (Conf: 0.9327) - File: D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\512Normal\normal (51).jpg
- True: no_tumor, Pred: glioma (Conf: 0.9319) - File: D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\512Normal\normal (92).jpg
- True: no_tumor, Pred: glioma (Conf: 0.9312) - File: D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\512Normal\normal (29).jpg
- True: no_tumor, Pred: glioma (Conf: 0.9311) - File: D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\512Normal\normal (37).jpg
- True: no_tumor, Pred: glioma (Conf: 0.9298) - File: D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw\512Normal\normal (85).jpg
