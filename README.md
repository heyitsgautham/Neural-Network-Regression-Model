# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective is to build and train a neural network for regression using PyTorch. A dataset containing input-output pairs is preprocessed, normalized, and split into training and test sets. A neural network with fully connected layers is designed and trained using backpropagation. The goal is to predict continuous target values based on input features.

## Neural Network Model

![image](https://github.com/user-attachments/assets/3f151a6f-df43-4ff8-9326-07f28c8382b7)


## DESIGN STEPS

### STEP 1:

Loading the dataset

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

## PROGRAM

### Name: GAUTHAM KRISHNA S
### Register Number: 212223240036
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,10)
        self.fc2=nn.Linear(10,12)
        self.fc3=nn.Linear(12,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}


  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x






# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')




```
## Dataset Information

![image](https://github.com/user-attachments/assets/9cecfe9d-5d80-4a60-8b5d-499348fb82cb)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/1562c261-804c-411b-a6e2-b51cb2e50933)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/e7a59b1c-0f0b-48eb-a9e6-d5644ff5dc89)


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
