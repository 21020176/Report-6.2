import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse

parser = argparse.ArgumentParser("ESFPNet based model")
parser.add_argument('--model', type=str, required=True,
        help='Type of model training')
parser.add_argument('--dataset', type=str, required=True,
        help='Location of the dataset')
args = parser.parse_args()


df = pd.read_csv(args.dataset)
df.drop(columns=["folder"],axis=1,inplace=True)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

if args.model == 'lr':
    model=LogisticRegression()
elif args.model == 'rf' :
    model = RandomForestClassifier()
elif args.model == 'mlp' :
    model = MLPClassifier(hidden_layer_sizes=(64, 32),
                        max_iter=1000, random_state=42)
elif args.model == 'svm' :   
    model = svm.SVC(kernel='linear')

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
recall = tp / (tp+fn)
print(accuracy_score(y_test, y_pred), "accuracy")
print(specificity, "specificity")
print(recall, "recall")