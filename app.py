# save this as app.py
from flask import Flask
from flask import request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})


@app.route('/')
def hello():
    return "Hello World"

@app.route('/check', methods = ['POST'])
def checkContent():
    check = request.get_data() 
    app.logger.warning(check)
    return getResult(check)

def getResult(check):
    with open('SpamClassifier.pkl', 'rb') as f:
        X, cv, clf = pickle.load(f)    
        data = [check]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        app.logger.warning(my_prediction[0])
        return str(my_prediction[0])


if __name__ == "__main__":
    app.run(debug=True)

app.run(port=5000)