# Coding-In-Cardiology-Challenge
This project is a machine learning model created to estimate the mortality rate of patients based on time-series data in order to allocate the proper resources to patients who need it most.

A logistic regression model was used to train a set of over 50 features, along with k-fold clustering to find the best regularization parameters to use. The model was then evaluated using a confusion matrix, on metrics such as accuracy and precision.

This model was coded using Python, utilizing packages such as Sklearn and Numpy to implement data aggregation and evaluation functions.

The model achieved an AUROC score of 0.85 with L2 regularization, using the optimal parameters found and the aggregated feature set. So, the model was able to do a good job of separating patients who were high risk versus those who were not.

To see this project, run hw3_challenge.py to see the performance of this main model. This project also included visualizations of the differences of L0, L1, and L2 regularization, which can be viewed by running hw3_main.py.

This allowed me to learn a lot about machine learning models and what goes into their creation. I learned about the design process of feature sets, how that is done, and how different parameters can be selected through different processes, in order to create a full model that is able to achieve good scores for this challenge.

<img width="1267" height="346" alt="image" src="https://github.com/user-attachments/assets/62ff68f9-66c3-44fe-bb36-09ddc154abfe" />

View more information here: <a target="_blank">https://miguelchaveznava.github.io/challenge.html<a>
