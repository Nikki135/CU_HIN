import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score
#from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# Machine Learning Packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

urls_data = pd.read_csv("urldata.csv")
#dns_data = pd.read_csv("destip2domain.csv")
#======= Getting our reverse DNS data ======================
dns_data =  pd.read_csv("destdomaindata.csv", sep=',', header=0,
                            names=['ip_addr','dns_name'], engine='python')
#Dup_Rows = dns_data[dns_data.duplicated()]
#DF_RM_DUP = dns_data.drop_duplicates(keep='first')
#print('\n\nResult DataFrame after duplicate removal :\n', DF_RM_DUP.head(n=5))
#print("\n\nDuplicate Rows : \n {}".format(Dup_Rows))
#print("Keys", dns_data.keys())
#print(dns_data['dns_name'])
dns = dns_data['dns_name'].values.tolist()
ipaddr = dns_data['ip_addr'].values.tolist()
ip_to_domain = dns_data.to_numpy()

#print("Reverse DNS data:", ip_to_domain[0][0], ip_to_domain[0][1])
def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/') # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-') # make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.') # make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens)) #remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com') #removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

# Labels
y = urls_data["label"]
#print(y)
# Feature extraction
url_list = urls_data["url"]

# Using Default Tokenizer
#vectorizer = TfidfVectorizer()

# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
# Store vectors into X variable as Our XFeatures
#print(url_list)
X = vectorizer.fit_transform(url_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_test)
# Model Building
#using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)

# Accuracy of Our Model
print("Accuracy ",logit.score(X_test, y_test))

X_predict = dns


x_predict=X_predict
#                for key in destip2domain:
#                    ip_to_domain.append([key, destip2domain[key]])
#                    print("IP_to_Domain matrix", ip_to_domain)
X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)
#print(New_predict)
n1_predict=vectorizer.fit_transform(New_predict)
#print("X_predict", n_predict)
ben_mal_matrix = []
#n2=y_test.to_numpy()
#n1=y_tra[:83588].to_numpy()
n1=np.random.choice(y_train, size=83588, replace=False)
#n1 = n2[n, :]
#data = { 'X_test', X_predict}
print(n1)
#print(New_predict)

for list_entry in range(len(New_predict)):
    if list_entry=="bad":
        New_predict[list_entry] = x_predict[list_entry] + " is Malicious"
        ben_mal_matrix.append('bad')
    else:
        New_predict[list_entry] = x_predict[list_entry] + " is Benign"
        ben_mal_matrix.append('good')

#y_tes1=vectorizer.fit_transform(y_test[:83588])
#n_predict=vectorizer.fit_transform(ben_mal_matrix)
#print(y_tes1)
#print(n_predict)
#df = pd.DataFrame((y_tes1).all(),(n_predict).all(), columns=['Y_test','Y_Predict'])
#confusion_matrix = confusion_matrix(df['Y_test'], df['Y_Predict'])
#confusion_matrix.print_stats()
#================== Need to append this information to our existing CSV file ======
#print(ben_mal_matrix)
##print("\n")
#for i in New_predict:
#    print(i)
print(accuracy_score(n1,ben_mal_matrix))
print(precision_score(n1,ben_mal_matrix, average='macro'))
print(recall_score(n1,ben_mal_matrix, average='macro'))
print(f1_score(n1,ben_mal_matrix, average='macro'))
#df = pd.DataFrame(n1,New_predict, columns=['y_test','New_Predict'])
#confusion_matrix = confusion_matrix(y_true=n1, y_pred=ben_mal_matrix)
##
#print(confusion_matrix)
#fig, ax = plt.subplots(figsize=(5, 5))
#ax.matshow(confusion_matrix, cmap=plt.cm.Oranges, alpha=0.3)
#for i in range(confusion_matrix.shape[0]):
#    for j in range(confusion_matrix.shape[1]):
#        ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center')
#
#plt.xlabel('Predicted label')
#plt.ylabel('True label')
#plt.title('Confusion Matrix')
#plt.show()
