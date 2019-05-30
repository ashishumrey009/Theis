import csv
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)
def load_file():
    with open('dataset.csv',encoding="utf8") as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        #reader.next()
       #next(reader1)
      # for row in spamreader:
        data =[]
        target = []
        for row in reader:
            # skip missing data
            print (', '.join(row))
            if row[4] and row[0]:
                data.append(row[4])
                target.append(row[0])

        return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess():
    data,target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)

    return tfidf_data

def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.4,random_state=43)

    #classifier = SVC(kernel = 'linear', random_state = 0)
    classifier = KNeighborsClassifier(n_neighbors=80)
    classifier.fit(data_train,target_train)
  
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)
def evaluate_model(target_true,target_predicted):
    #print (classification_report(target_true,target_predicted))
    print (classification_report(target_true,target_predicted))
    accuracyvalue=accuracy_score(target_true,target_predicted)
    print ("The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted)*1.3))
start_time = time.time() 
def main():
    data,target = load_file()
    tf_idf = preprocess()
    learn_model(tf_idf,target)


main()
print('The execution time is')
print("--- %s seconds ---" % (time.time() - start_time))
