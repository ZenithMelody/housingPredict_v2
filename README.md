# 🏘️ [Iteration of previous housing prediction project](https://github.com/ZenithMelody/logisticRegression_scikit) + now with Hyperparameter Tuning capabilites 

## ✨ What does it do?
* **Problem:** Using the dataset it will predict if a flat is big (5 and above) or not (4 rooms or less)
* **Finding:** Price is a massive predictor of Size
* **Upgrades:** Diagnostic mode (allows to select either to the best model or manual tuning of model)
* **Accuracy:** 80%

## 🚩 Issues & Solution
* **Reading:** Accuracy did not improve?
* **Fix:** It hit the "Bayes Error Rate" = lowest possible theoretical error rate that a classifier can achieve on a given classification problem (which meant this repo is spent learning to break the model to see what each tuning does)

## 🛠️ Tech Stack
* **Language:** Python 3.11/9
* **Library:** 
  * `scikit-learn`
  * `pandas` and `numpy` (data manipulation)
  * `matplotlib` and `seaborn` (data visualization)
  * `GridSearchCV` (makes the tuning possible)
* **Algorithm:** Logistic Regression

## 🚀 How to Run

### Prerequisites
Ensure you have Python installed. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## ⚙️Model Settings & What they do?
| Setting | Name | Effect |
| ------------- | ------------- | ------------- |
| `C=0.000001` | Inverse Regularization Strength - a "leash" that pulls the model's cofficients towards zero | a tiny `C` forces the model to underfit, without a higher `C`. the model becomes "lazy" and fails to capture the S-Curve |
| `class_weight={0: 1, 1: 1000}` | Cost-Sensitive Learning | Scoring rules, usually mistakes = 1 point but here its 1000, shows that we can force the model to prioritize "recall" (find all "Big Flats") over "precision" (being right) |
| `max_iter=5000` | Solver Iteration Limit | give the model a "budget of steps" to find the solution, which ensures the model does not give up early, e.g if setttings are terrible it forces the model to try for the best solution within the budget |
| `penalty='l2'` | Regularization | prevents model from overfitting. `l2` is the standard default, shrinks all weights evenly without deleting - in contrast to `l1` which deletes useless features entirely by setting them to zero |
| `solver=lbfgs` | Limited-memory Broyden-Goldfarb-Shanno | find the lowest error - it is the default in `scikit-learn` due to its fast and memory-efficient performance for small-medium datasets |
  
## 📊 Visuals (Model Performance)
| Best Model | Diagnostic | 
| ------------- | ------------- |
| <img width="251" height="108" alt="{9EFE624E-5342-4288-B667-59BCFEB50351}" src="https://github.com/user-attachments/assets/1f488106-278d-477b-a522-30986c08043c" />  | <img width="239" height="106" alt="{74106E9A-6B5A-478B-AF7E-0BB613E12D49}" src="https://github.com/user-attachments/assets/6bacd27a-127d-47e4-823b-187975a14036" /> |

## 📊 Visuals (Graph and Curve)
❗Note: S-Curve helps to verify data relationship, but please refer to the Confusion Matrix for the model's actual performance - thus even in the diagnostic/stress mode the curves look similar due to the underlying strong data trend 
| Diagnostic Mode  | Confusion Matrix | Curve | 
| ------------- | ------------- | ------------- |
| Disabled (Best Model) | <img width="1111" height="1215" alt="{D078E220-D754-4CC6-B351-1F3370EA842A}" src="https://github.com/user-attachments/assets/2f4e0de5-db88-4c4e-b480-74c84873e0e2" /> | <img width="1091" height="1167" alt="{FF28B028-D0BB-41B4-89B5-769D14BDC547}" src="https://github.com/user-attachments/assets/d546bb50-624b-4cd5-b153-35ddcfffa1df" /> |
| Enabled (Stress Test) | <img width="1073" height="1185" alt="{9F455383-4B73-4819-90CD-85AEFA4E543A}" src="https://github.com/user-attachments/assets/850f70fd-0713-43cc-93e9-575d02fd3175" /> | <img width="1056" height="1137" alt="{B3B36A1C-AD71-4585-9405-D4D21A8221CD}" src="https://github.com/user-attachments/assets/1de5962d-14f2-4b1a-91b8-a45db64fd4c9" /> |
