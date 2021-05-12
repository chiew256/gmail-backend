import pickle

with open('SpamClassifier.pkl', 'rb') as f:
    X, cv, clf = pickle.load(f)    
    message = input('enter the email: ')
    while(True):
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction[0])
        message = input('enter the email: ')