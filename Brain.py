import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
filename = 'excel.xlsx'
skiprows = 2
header_row = 1
alpha_range = (8, 12)
beta_range = (12, 30)
delta_range = (0.5, 4)
theta_range = (4, 8)
def classify_eeg_data(filename, skiprows, header_row, alpha_range, beta_range, delta_range, theta_range):
    # Load the EEG data from Excel sheet
    df = pd.read_excel(filename, skiprows=skiprows, header=header_row)
    df.columns = df.columns.astype(int)
    # Remove non-numeric characters from column headers and convert them to strings
    df.columns = df.columns.astype(str).str.replace('[^0-9\.]', '')
    # Convert column headers to integers
    df.columns = df.columns.astype(int)
    fft_values = df.iloc[:, 1:].values
    # Extract power spectral density features from FFT values
    alpha_waveform = np.fft.irfft(np.where((df.columns >= alpha_range[0]) & (df.columns <= alpha_range[1]), fft_values, 0), axis=1)
    beta_waveform = np.fft.irfft(np.where((df.columns >= beta_range[0]) & (df.columns <= beta_range[1]), fft_values, 0), axis=1)
    delta_waveform = np.fft.irfft(np.where((df.columns >= delta_range[0]) & (df.columns <= delta_range[1]), fft_values, 0), axis=1)
    theta_waveform = np.fft.irfft(np.where((df.columns >= theta_range[0]) & (df.columns <= theta_range[1]), fft_values, 0), axis=1)
    alpha_power = np.sum(df.iloc[:, (df.columns >= alpha_range[0]) & (df.columns <= alpha_range[1])].values, axis=1)
    beta_power = np.sum(df.iloc[:, (df.columns >= beta_range[0]) & (df.columns <= beta_range[1])].values, axis=1)
    delta_power = np.sum(df.iloc[:, (df.columns >= delta_range[0]) & (df.columns <= delta_range[1])].values, axis=1)
    theta_power = np.sum(df.iloc[:, (df.columns >= theta_range[0]) & (df.columns <= theta_range[1])].values, axis=1)
    # Combine the features into a feature matrix
    X = np.column_stack((alpha_power, beta_power, delta_power, theta_power))
    # Define the labels for the classification task
    y = np.array(['alpha' if i < 40 else 'beta' if i < 80 else 'delta' if i < 120 else 'theta' for i in range(df.shape[0])])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Standardize the feature values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Train an Artificial Neural Network (ANN) classifier
    classifier = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', solver='adam', max_iter=1000, random_state=0)
    classifier.fit(X_train, y_train)
    # Predict the labels of the test set
    y_pred = classifier.predict(X_test)
    # Evaluate the performance of the classifier
    cm = confusion_matrix(y_test,y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel='Predicted label', ylabel='True label')
    # Loop over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    plt.show()
    # Print the performance metrics
    print('Accuracy: {:.2f}%'.format(acc * 100))
    print('Precision: {:.2f}%'.format(prec * 100))
    print('Recall: {:.2f}%'.format(rec * 100))
    print('F1-score: {:.2f}%'.format(f1 * 100))
classify_eeg_data(filename, skiprows, header_row, alpha_range, beta_range, delta_range, theta_range)
