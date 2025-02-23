# Project Chrysalis: ECoG_motor_imagery

## Abstracts

Brain-Computer Interface (BCI), based on motor imagery, translates the motor intention into a control signal by classifying the electrophysiological patterns of different imagination tasks using ECoG, which can capture a broader range of frequencies showing better sensitivity and higher input quality than EEG. However, with ECoG being an invasive technique, there may be some utility in developing an ECoG bidirectional classifier that reduces the number of implanted electrodes.

The present study aims to develop an ECoG classifier to achieve high accuracy with a limited number of electrodes. We will work with ECoG datasets from Miller 2019, recorded in a clinical setting with motor and imagery tasks from seven subjects.

For data pre-processing, we will start with artifact removal and then apply a bandpass filter with the frequency band of 8-30 Hz. We will use event-related desynchronization (ERD) and readiness potential (RP) as features. In addition, we will divide the dataset into different epochs corresponding to the different imagery trials in the experiment. For classification, we will use Support Vector Machines (SVM), Linear Discriminant Analysis (LDA), and k Nearest Neighbor (kNN) classifiers. First, we plan to increase the accuracy of our models using cross-validation; after that, we will measure the accuracy of predictions for each model using the root mean square error (RMSE) and confusion matrix.

Previous investigations into applying LDA, KNN, and SVM as classifiers for tongue vs finger movement have yielded results with 60%, 87%, and 83% accuracy. We predict that the models will maintain their utility as classifiers with similar accuracies in classifying input data as either tongue or hand imagery movement as we reduce the number of electrodes. Developing a classification algorithm for different imagery signals will be a milestone in developing an ECoG-based BCI. Furthermore, reducing the electrodes will minimize the invasiveness in future BCI applications.

## References

Miller, K. J., Schalk, G., Fetz, E. E., den Nijs, M., Ojemann, J. G., & Rao, R. P. N. (2010). Cortical activity during motor execution, motor imagery, and imagery-based online feedback. Proceedings of the National Academy of Sciences, 107(9), 4430–4435. [https://doi.org/10.1073/pnas.0913697107](https://doi.org/10.1073/pnas.0913697107)

Chong, L., Zhao, H. B., Li, C. S. and Hong, W. (2010) 'Classification of ECoG signals for motor imagery tasks,' 2010 2nd International Conference on Signal Processing Systems. Dalian, P. R. China, 2010.IEEE.


#### [Video Recording](https://www.world-wide.org/neuromatch-5.0/classifying-motor-imagery-ecog-signal-bb5dedd9/) 
