import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score

#Reading the file with train data and adding a label column
df = pd.read_csv('trainData/42_train.csv', header=None)
df['label'] = 0
df.loc[2000:, 'label'] = 1

#Reading the file with test data
test_data = pd.read_csv('trainData/42_test.csv', header=None)

#Splitting dataset into features and target variable
X = df.iloc[:, :-1].values  #Convert to numpy array bc of pytorch
y = df.iloc[:, -1].values

#pytorch neural network architecture
class CustomNet(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, 64)  #Increased hidden layer size for better capacity
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  #Output layer
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))  #First hidden layer with ReLU
        x = self.act(self.fc2(x))  #Second hidden layer with ReLU
        x = self.fc3(x)  #No activation here for final output (it's a single unit bc we want a single probability)
        return x

#Check for CUDA device and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #if CUDA available we can perform neural networks much faster by they are happening simultaneously

#Parameters
n_folds = 5
kn = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
train_epochs = 1000
n_features = X.shape[1]

#Cross-validation loop
fold_aucs = []
plt.figure(figsize=(8, 6))  #Adjust plot size
for fold, (train_idx, val_idx) in enumerate(kn.split(X, y), 1):     #For each fold we calculate ROC and AUC
    #Prepare training and validation data for the current fold
    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    #Initialize model, loss function, and optimizer
    model = CustomNet(n_features).to(device)
    loss_fn = nn.BCEWithLogitsLoss()  #Binary cross-entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   #Optimizer with learning rate 1e-3 :)

    #Training the model
    model.train()
    for epoch in range(train_epochs):
        optimizer.zero_grad()  #Reset gradients
        logits = model(X_train).squeeze(1)  #Forward pass
        loss = loss_fn(logits, y_train)  #Calculate loss
        loss.backward()  #Backpropagate the error
        optimizer.step()  #Update model weights

    #Model evaluation on validation data
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val).squeeze(1)  #Get raw logits from the model
        val_probs = torch.sigmoid(val_logits).cpu().numpy()  #Apply sigmoid for probabilities
        y_val_np = y_val.cpu().numpy().astype(int)

    #ROC curve and AUC calculation
    auc = roc_auc_score(y_val_np, val_probs)
    fpr, tpr, thresholds = roc_curve(y_val_np, val_probs)
    fold_aucs.append(auc)

    # Plotting ROC curve for each fold
    plt.plot(fpr, tpr, alpha=0.6, label=f'Fold {fold} AUC={auc:.4f}')

#Final reporting
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
print(f'Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}')

#Plot ROC curve across all folds
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves per fold')
plt.legend()
plt.show()

#Training on full dataset (for final predictions)
#Prepare full training data with re-weighting
w0 = 0.4 / 0.5
w1 = 0.6 / 0.5
w_full = np.where(y == 0, w0, w1)

#Final model training on the whole dataset
X_full = torch.tensor(X, dtype=torch.float32, device=device)
y_full = torch.tensor(y, dtype=torch.float32, device=device)

final_model = CustomNet(n_features).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)
final_loss_fn = nn.BCEWithLogitsLoss()      #binary cross-entropy with logits combines a sigmoid activation with the binary cross-entropy loss

#Train final model
final_model.train()
for epoch in range(train_epochs):
    final_optimizer.zero_grad()
    logits = final_model(X_full).squeeze(1)
    loss = final_loss_fn(logits, y_full)
    loss.backward()
    final_optimizer.step()

#Making predictions on test data
X_test = torch.tensor(test_data.values, dtype=torch.float32, device=device)
final_model.eval()
with torch.no_grad():
    test_logits = final_model(X_test).squeeze(1)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()

#Save predictions to CSV
submission = pd.DataFrame(test_probs)
submission.to_csv('42_submission_neural_network.csv', index=False, header=False)
print("Submission shape should be (50000,):", test_probs.shape)

