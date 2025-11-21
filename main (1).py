import numpy as np
from scipy.io import loadmat
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


 #  parse/loading and normalize dataset
def loadData(path):

    data = loadmat(path)
    image_data = data['X']  # shaping the data
    labels = data['y'].flatten()  # shaping the data
    labels[labels == 10] = 0  # Replace '10' with digit '0'

    # I tried to run this without normalizing the data and it took about 10 minutes for the model to finish. 
    # Normalizing drastically changes the speed.
    # Pixel values range from 0 to 255, so we normalize to [0, 1] to help gradient descent converge faster.
    image_data = np.transpose(image_data, (3, 0, 1, 2))
    flattened_images = image_data.reshape(image_data.shape[0], -1) / 255.0 # <- normalization

    return flattened_images, labels

# Load SVHN dataset from .mat files using load_svnh_data
# The data is loaded into: 
# trainingValueX: flattened and normalized image pixels for training
# trainingValueY: digit labels (0–9) for training
# testingValueX: image data for testing
# testingValueY: digit labels for testing
trainingValueX, trainingValueY = loadData('train_32x32.mat')
testingValueX, testingValueY = loadData('test_32x32.mat')

def report(y_true, y_pred, model_name="Model"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)
    print(f"\n{model_name} Report For Model to properly view the data:\n")
    print(df.to_markdown())

# Naive Bayes Classifier
# Uses GaussianNB from sklearn assumes pixel features follow a normal distribution
# Evaluates accuracy and prints a classification report
print("\nUsing Naive Bayes\nEstimating mean and variance for each pixel feature per digit class\nApplying Bayes' Theorem to calculate probabilities\n")
model1 = GaussianNB()
model1.fit(trainingValueX, trainingValueY)
model1_prediction = model1.predict(testingValueX)
print(f"Naive Bayes Predicition values: {model1_prediction}")
print("Naive Bayes Results:")
accuracy1 = accuracy_score(testingValueY, model1_prediction) * 100
print(f"Naive Bayes total model accuracy = {accuracy1}%")
report(testingValueY, model1_prediction, model_name="Naive Bayes")

# Logistic Regression with PCA
#  PCA (Principal Component Analysis) reduces the number of features from 3072 to 100 shown below
#  logistic regression trains on PCA-reduced data
#  logistic regression to distinguish digits
print("\nUsing gradient descent\nadjusting the weights \n")
pca = PCA(n_components=100) 
trainingValueX_pca = pca.fit_transform(trainingValueX)
testingValueX_pca = pca.transform(testingValueX)
model2 = LogisticRegression(max_iter=1000, solver='lbfgs')
model2.fit(trainingValueX_pca, trainingValueY)
model2_prediction = model2.predict(testingValueX_pca)
print(f"Logistic Regression Predicition values: {model2_prediction}")
print("Logistic Regression Results:")
accuracy2 = accuracy_score(testingValueY, model2_prediction) * 100
print(f"Logistic regression total model accuracy = {accuracy2}%")
report(testingValueY, model2_prediction, model_name="Naive Bayes")

####################################################################################
####################################################################################
# This is where the nueral network training begins

X_train_tensor = torch.tensor(trainingValueX, dtype=torch.float32)
y_train_tensor = torch.tensor(trainingValueY, dtype=torch.long)
X_test_tensor = torch.tensor(testingValueX, dtype=torch.float32)
y_test_tensor = torch.tensor(testingValueY, dtype=torch.long)

# Loading the data 
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Sigmoid is used to push the values inbetween 0-1 
# We define a feedforward network with a few hidden layers and a sigmoid activation function
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32*32*3, 256),  
            nn.Sigmoid(),             
            nn.Linear(256, 128),      
            nn.Sigmoid(),            
            nn.Linear(128, 10)       
        )

    def forward(self, x):
        return self.model(x)


# CrossEntropy is used for the softmax function
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
# For the data we backpropagate though 10 times, or do 10 runs through the data to make the model more accurate, this number can
# change but slows down the speed of the running the process
# calculate the error and do forward passes through the model
print("Using nueral networks to calculate accuracy on model\n")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()             
        outputs = model(X_batch)           
        loss = criterion(outputs, y_batch) 
        loss.backward()                    
        optimizer.step()                   
        total_loss += loss.item()          
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss}")

# find the inference of the model
# Use torch.max to get the predicted class from output scores
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

# Print out final accuracy and classification report
nn_accuracy = accuracy_score(all_labels, all_preds) * 100
print(f"\nNeural Network with Sigmoid total model accuracy = {nn_accuracy}%")
report(all_labels, all_preds, model_name="Neural Network (Sigmoid)")

print(f"\nNeural Network with Sigmoid total model accuracy = {nn_accuracy}%")
print(f"Logistic regression total model accuracy = {accuracy2}%")
print(f"Naive Bayes total model accuracy = {accuracy1}%")

