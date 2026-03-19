# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Design a neural network model to solve a regression problem using a single input feature.
The model uses multiple fully connected layers with ReLU activation to predict a continuous output value.
Train the network using a loss function and optimizer over several epochs to minimize error.
Track and display the training loss during the learning process.


## Neural Network Model

<img width="1313" height="768" alt="Screenshot 2026-02-02 094500" src="https://github.com/user-attachments/assets/808b6fd6-4f00-4a68-b04b-24b6082ab34d" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Rithika R

### Register Number:212224240136

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```python
dataset1 = pd.read_csv('rit-1.csv')
X = dataset1[['input']].values
y = dataset1[['output']].values
```
```python
print(dataset1.head())
```
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
```
```python
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```
```python
# Name:Rithika R
# Register Number:212224240136
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x       
```
```python
# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.001)#lr=learning rate
```
```python
# Name:Rithika R
# Register Number:212224240136
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad() 
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
```python
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```python
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
```
```python
loss_df = pd.DataFrame(ai_brain.history)
```
```python
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
```
```python
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="331" height="297" alt="gr" src="https://github.com/user-attachments/assets/0eb5731f-840c-4d64-9878-c05bd18eef62" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/74c9ced3-4bc7-476d-87a9-8059bbd83af2" />



### New Sample Data Prediction
<img width="1029" height="294" alt="image" src="https://github.com/user-attachments/assets/5e577808-874d-4c44-9574-448c9e4829d5" />


<img width="893" height="133" alt="image" src="https://github.com/user-attachments/assets/e689a0d0-b264-4718-8ef1-046b085a87a5" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
