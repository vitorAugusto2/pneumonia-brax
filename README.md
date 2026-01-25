
██████╗ ██████╗  █████╗ ██╗  ██╗
██╔══██╗██╔══██╗██╔══██╗╚██╗██╔╝
██████╔╝██████╔╝███████║ ╚███╔╝
██╔══██╗██╔══██╗██╔══██║ ██╔██╗
██████╔╝██║  ██║██║  ██║██╔╝ ██╗
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

      P N E U M O N I A

Checking dependencies...

Running...
DISTRIBUTION OF PATIENTS BY GROUP
Patients with pneumonia      = 380
Other diseases + No Findings = 18062

TRAIN
target
0    373
1    266

VAL
target
0    104
1     76

TEST
target
0    45
1    38

=> VGG16 (+) RF

EVALUATION METRICS
VAL
Accuracy  = 0.669
Precision = 0.600
Recall    = 0.212
F1-Score  = 0.313
AUC       = 0.658
Confusion Matrix
 [[260  22]
 [123  33]]

TEST
Accuracy  = 0.655
Precision = 0.667
Recall    = 0.300
F1-Score  = 0.414
AUC       = 0.763
Confusion Matrix
 [[105  12]
 [ 56  24]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 00:02
Total              = 00:45

=> VGG16 (+) XGB

EVALUATION METRICS
VAL
Accuracy  = 0.655
Precision = 0.521
Recall    = 0.404
F1-Score  = 0.455
AUC       = 0.642
Confusion Matrix
 [[224  58]
 [ 93  63]]

TEST
Accuracy  = 0.695
Precision = 0.652
Recall    = 0.537
F1-Score  = 0.589
AUC       = 0.757
Confusion Matrix
 [[94 23]
 [37 43]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 01:22
Total              = 02:05

=> VGG16 (+) SVM

EVALUATION METRICS
VAL
Accuracy  = 0.685
Precision = 0.562
Recall    = 0.526
F1-Score  = 0.543
AUC       = 0.658
Confusion Matrix
 [[218  64]
 [ 74  82]]

TEST
Accuracy  = 0.670
Precision = 0.593
Recall    = 0.600
F1-Score  = 0.596
AUC       = 0.741
Confusion Matrix
 [[84 33]
 [32 48]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 00:21
Total              = 01:04

=> RESNET50 (+) RF

EVALUATION METRICS
VAL
Accuracy  = 0.689
Precision = 0.656
Recall    = 0.269
F1-Score  = 0.382
AUC       = 0.704
Confusion Matrix
 [[260  22]
 [114  42]]

TEST
Accuracy  = 0.695
Precision = 0.750
Recall    = 0.375
F1-Score  = 0.500
AUC       = 0.742
Confusion Matrix
 [[107  10]
 [ 50  30]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 00:03
Total              = 00:47

=> RESNET50 (+) XGB

EVALUATION METRICS
VAL
Accuracy  = 0.678
Precision = 0.568
Recall    = 0.404
F1-Score  = 0.472
AUC       = 0.722
Confusion Matrix
 [[234  48]
 [ 93  63]]

TEST
Accuracy  = 0.690
Precision = 0.656
Recall    = 0.500
F1-Score  = 0.567
AUC       = 0.757
Confusion Matrix
 [[96 21]
 [40 40]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 01:18
Total              = 02:02

=> RESNET50 (+) SVM

EVALUATION METRICS
VAL
Accuracy  = 0.626
Precision = 0.476
Recall    = 0.506
F1-Score  = 0.491
AUC       = 0.657
Confusion Matrix
 [[195  87]
 [ 77  79]]

TEST
Accuracy  = 0.665
Precision = 0.583
Recall    = 0.613
F1-Score  = 0.598
AUC       = 0.721
Confusion Matrix
 [[82 35]
 [31 49]]

TIME RUN (MM:SS)
Feature extraction = 00:43
Training           = 00:09
Total              = 00:52
