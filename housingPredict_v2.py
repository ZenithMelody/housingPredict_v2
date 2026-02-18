import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV # new tool
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# --- Data ---
df = pd.read_csv('data.csv')

# --- Classification ---
big_flats = ['5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
# Define what counts as "big"
df['is_Big_Flat'] = df['flat_type'].apply(lambda x: 1 if x in big_flats else 0)

X = df[['resale_price']]
y = df['is_Big_Flat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Hyperparameters ---
# 1. Define the "Hyperparameter Grid" (The Menu of Options)
# We test different "C" values (Strengths) and "penalty" types (Rules)
# Note: 'lbfgs' solver only supports 'l2' or None. 'liblinear' supports 'l1'.
param_grid = [
    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}, 
    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear']} 
]

# Base model
base_model = LogisticRegression(max_iter=5000)

# --- Grid Search ---
# cv=5 means "Cross Validation": Spliting data into 5 chunks to double-check results.
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', verbose=1)

# --- Train Grid Search ---
print("Traning in progress, please wait.")
grid_search.fit(X_train, y_train)

# --- Result ---
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best training accuracy: {grid_search.best_score_:.2%}")

# --- Evaluation ---
print("-" * 40)
option_Model = input("Enable diagnostic mode? (Y/N): ").upper()

if option_Model == "Y":
    print("Diagnostic mode -  for experimentation purposes")
    # to find outliers - e.g Tiny C + Extreme Bias
    best_model = LogisticRegression(C=0.000001, class_weight={0: 1, 1: 1000}, max_iter=5000)
    diagnostic = True 
elif "N":
    print("Best performance mode - for production use")
    # from your Grid Search results
    best_model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=5000)
    diagnostic = False
else:
    print("Invalid input, please enter either Y or N")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print("--- Model Performance ---")
print(f"Is Diagnostic Mode enabled? {diagnostic}")
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.2%}")
print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_test, y_pred)}")
print("-" * 40)

# --- Confusion Matrix ---
plt.figure(figsize=(6, 5))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f"Best Model (C={grid_search.best_params_['C']})")
plt.xlabel('Predicted (0=Not Big, 1=Big)')
plt.ylabel('Actual (0=Not Big, 1=Big)')
plt.show()


# --- Probability Curve ---
plt.figure(figsize=(8, 6))
sns.regplot(x=df['resale_price'], y=df['is_Big_Flat'], logistic=True, ci=None, scatter_kws={'color': 'black'}, line_kws={'color': 'red'})
plt.title('Probability of being a big flat vs. Price')
plt.xlabel('Price')
plt.ylabel('Probability of big flat (0 to 1)')
plt.show()