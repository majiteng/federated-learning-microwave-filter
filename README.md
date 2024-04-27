# federated-learning-microwave-filter
Code and related dataset for the paper "Federated Learning for Microwave Filter Behavior Prediction", which has been published by IEEE Microwave and Wireless Technology Letters.
The manuscript can be found here: https://ieeexplore.ieee.org/abstract/document/10411954.

Abstract:
Deep learning (DL) technologies have been widely investigated to improve the performance of microwave device behavior prediction. Advanced microwave-related DL technologies utilize independent computers to collect data from the electronic design automation (EDA) software. However, it is essential to note that DL requires a vast amount of high-quality training data. Collecting these data from exact simulation meticulous optimization in EDA is exceptionally time-consuming and computationally intensive. A straightforward way to speed up the process is by collecting quality data from distributed RF designers. However, this approach may not always be feasible due to the need to maintain the confidentiality of sensitive microwave design information. In this letter, we proposed a federated learning (FL) framework for corporately training DL models for microwave filter behavior prediction. The FL framework aggregates knowledge from various designers without sharing their raw data. The primary experimental results demonstrate the feasibility of the proposed encrypted FL framework for microwave filter applications with superior accuracy and speed.

The datasets contain two examples, including a stepped-impedance low-pass filter (SIF) in the folder "Data_SIF" and a hairpin bandpass filter (HPF) in the folder "Data_HPF".

In the folder "stepped_impedance/Data", there are three different SIF types: 4th-, 5th-, and 6th-order SIFs. In each filter type, there are three .csv files representing numbers with different precision.
In each .csv file, labels "W1-W6, L1-L6" are the geometry of the filter. The rest of the labels are corresponding behaviors, including S21M (real part of S21) and S21P (imaginary part of S21). The frequency range is from 0.1GHz to 6.0GHz with an interval of 400MHz.

In the folder "hair_pin/Data", there are three different types of HPF, including 3rd-, 5th-, and 7th-order HPF. In each filter type, there are three .csv files representing numbers with different precision.
In each .csv file, labels "W1-W3, L1-L5, S1" are the geometry of the filter. The rest of the labels are corresponding behaviors, including S21M (real part of S21) and S21P (imaginary part of S21). The frequency range is from 3.0GHz to 7.0GHz with an interval of 100MHz.
