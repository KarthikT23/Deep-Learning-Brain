# Deep-Learning-Brain
Analysis of Brain signals using Deep Learning Algorithms
# Types of Brain Waves
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/c80403c3-1c4d-47b9-b2fc-ee70eb667970)

# RMS Maximus 32 EEG and its amplitude map
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/336feb40-dc29-4e1a-b4dc-da399e87ab0d)
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/d2d98844-3102-440a-991b-a770c9cd8377)

# Feature extraction in EEG
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/65bc30df-646d-46e1-a713-5ad9eb16d418)

# Classification in EEG
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/7a4e3dd2-223b-4b0c-b3d8-85c1b8620f35)

# Design of Brain Computer Interface
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/2e1e8a9f-31c3-4f63-8bae-e53712a5d932)

# Flowchart of Methodology
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/6acb7be9-768f-4cb0-86f1-97ecdc316321)

# Block Diagram of BCI using EEG
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/d201e314-c19e-4659-a94e-96a6efb1e4bb)

# Setup
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/d6336224-abf7-4c4c-99ef-7238d4a5190c)

# Algorithm 
i. Import necessary libraries - pandas, numpy, sklearn, and matplotlib

ii. Define function 'classify_eeg_data' with parameters filename, skiprows, header_row, alpha_range, beta_range, delta_range, and theta_range

iii. Load EEG data from Excel sheet using pandas read_excel method, skip non-numeric characters from column headers, convert them to strings and then to integers

iv. Extract FFT values from loaded data and calculate power spectral density features

v. Combine the features into a feature matrix 'X' and define labels 'y' for classification task

vi. Split the data into training and testing sets

vii. Standardize the feature values using StandardScaler from sklearn

viii. Train a Multi-Layer Perceptron (MLP) Artificial Neural Network (ANN) classifier using MLPClassifier from sklearn, with hidden layer sizes (8, 4), 'relu' activation, 'adam' solver, maximum iterations of 1000, and a random state of 0

ix. Predict the labels of the test set using the trained classifier

x. Evaluate the performance of the classifier using confusion matrix, accuracy score, precision score, recall score, and F1 score from sklearn.metrics

xi. Plot the confusion matrix using matplotlib.pyplot.imshow

xii. Print the performance metrics - accuracy, precision, recall, and F1 score

xiii. Call the 'classify_eeg_data' function with appropriate parameters.

# Flowchart of Code
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/1992b4f4-16ee-4fa5-83bb-9a1104a50165)

# Before activity results
EEG Signal before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/c5e1dce8-8ad8-4fdb-9801-ae979e1adb68)

Feature Matrix and ANN parameters of FFT values calculated using No Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/8eb6497b-02da-4aca-a4d7-6cdd9f5f2fbc)


Frequency vs PSD of FFT values calculated using No Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/eed8584b-3555-4e4d-b20e-eb78b31b72c6)


Feature Matrix and ANN parameters of FFT values calculated using Blackman Window, before activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/1fae2c50-ccb7-4fb3-b601-16aa9bf97f4b)


Frequency vs PSD of FFT values calculated using Blackman Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/5338e84a-00f2-4d8d-9518-cb1da0c553b6)


Feature Matrix and ANN parameters of FFT values calculated using Hamming Window, before activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/67a2ff21-c9b0-48eb-9d06-2ec5522c38c2)


Frequency vs PSD of FFT values calculated using Hamming Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/7a7b3e92-2c21-45ba-9121-a9be023e5b43)


Feature Matrix and ANN parameters of FFT values calculated using Hanning Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/28060aac-11b2-4fa1-bdda-e8882a1586bc)



Frequency vs PSD of FFT values calculated using Hanning Window, before activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/1461cb79-93d1-4c22-96be-72b5080acdaf)

# After activity results
EEG Signal after activity



![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/80c52669-d74e-4055-8859-6aee63bdfd2a)


Feature Matrix and ANN parameters of FFT values calculated using No Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/9935d494-4f7a-47fc-9f49-50f768fdd418)


Frequency vs PSD of FFT values calculated using No Window, after activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/866a6c85-dd97-4615-bfa5-aafa5100c034)


Feature Matrix and ANN parameters of FFT values calculated using Blackman Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/91989da2-eb78-4b63-993a-8578c4d793b8)


Frequency vs PSD of FFT values calculated using Blackman Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/29a4189c-3e1e-42b0-bed3-db47b286954e)


Feature Matrix and ANN parameters of FFT values calculated using Hamming Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/4bc17ca9-59e4-4063-9b0b-661de546f3ab)


Frequency vs PSD of FFT values calculated using Hamming Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/94d4acec-f5af-4110-bd1f-2e18cd0b68dd)


Feature Matrix and ANN parameters of FFT values calculated using Hanning Window, after activity


![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/a5647402-aab0-4f0d-91dd-162da3639285)


Frequency vs PSD of FFT values calculated using Hanning Window, after activity

![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/cc08470e-3546-40ab-bc5f-368312bd8f26)

# Parameters before activity
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/7775dab3-6c16-4d3a-968c-6b3b5078675c)

# Parameters after activity
![image](https://github.com/KarthikT23/Deep-Learning-Brain/assets/119528503/6c731352-f891-4e6e-bae1-5e4dcae4035b)


# References
[1] Cecotti, H., & Graser, A. (2011). Convolutional neural networks for P300 detection with application to brain-computer interfaces. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(3), 433-445.

[2] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013.

[3] Roy, D., Banville, H., & Bengio, Y. (2019). Deep learning for electroencephalography: a review. arXiv preprint arXiv:1910.06331.

[4] Wu, C., Wu, X., Zhang, Y., Zhang, Y., & Wang, Y. (2020). Deep learning-based electroencephalography analysis: a review. Journal of Neuroscience Methods, 340, 108732.

[5] Lotte, F., Congedo, M., Lécuyer, A., Lamarche, F., & Arnaldi, B. (2007). A review of classification algorithms for EEG-based brain-computer interfaces. Journal of Neural Engineering, 4(2), R1-R13.

[6] Hosseini, M. P., Pascual-Marqui, R. D., & Niedermeyer, E. (2020). Deep learning in EEG and MEG: a review. Current Opinion in Neurology, 33(2), 198-207.

[7] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.

[8] Lawhern, V. J., Wu, W., Hatsopoulos, N. G., Paninski, L., & Sederberg, P. B. (2018). Deep learning for decoding human brain activity. Nature Reviews Neuroscience, 19(5), 327-339.

[9] Tabar, Y. R., Halici, U., & Atay, M. B. (2017). Automated diagnosis of Alzheimer's disease using deep neural networks. Computers in Biology and Medicine, 82, 32-40.
