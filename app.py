from flask import Flask,request,render_template
from preprocessing_pipeline import *

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



app = Flask('__main__')


@app.route('/')
def index():
    return render_template('index.html',result = None)


@app.route('/predict/',methods = ['post'])
def predict():
    
    input_ =  request.form['input']
    #preporcess the input 
    # input_ = demo_review
    model = request.form['model']
    print(model)
    output_ = proprocess_input(input_,mode = model)

    return render_template('index.html',result = output_)



if __name__ == "__main__":
    app.run()        