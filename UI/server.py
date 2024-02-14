from flask import Flask, request, render_template
import pickle
import numpy as np

# load the model
with open('./Insurance_premium_prediction_model.pkl', 'rb') as file:
      model = pickle.load(file)


# # create a flask application
app = Flask(__name__)
@app.route("/", methods=["GET"])
def root():
    # read the file contents and send them to client
    return render_template('index.html')

ra3 = None
@app.route("/Predict.html", methods=["POST"])
def predict():
    # get the values entered by user
    # print(request.form)

    age = request.form.get("Age_of_person")
    
    genderarr = [ 'Female' , 'Male']
    inpt = request.form.get("Gender_of_person")
    gender = int(genderarr.index(inpt))
    
    bmi = request.form.get("BMI_of_person")

    bloodpressure = int(request.form.get("Bloodpressure_of_person"))
    

    diabeticarr = [ 'No' , 'Yes']
    dinpt = request.form.get("Diabetes_of_person")
    diabetics  = int(diabeticarr.index(dinpt))
    
    children = int(request.form.get("No_of_childrens"))
    
    
    smokerarr = [  'No' , 'Yes']
    sinpt = request.form.get("Smoking_status")
    smoker  = int(smokerarr.index(sinpt))

    regionarr = ['r_northeast','r_northwest','r_southeast','r_southwest']
    region_onehoten = np.zeros(len(regionarr))
    rinpt = request.form.get("person_region")
    region_onehoten[regionarr.index(rinpt)] = 1
    

    answers = np.array([[age , gender , bmi , bloodpressure , diabetics , children , smoker  ]])
    region_onehoten = region_onehoten.reshape(1,-1)
    premium_result = np.concatenate((answers[:, :7], region_onehoten, answers[:,7:]), axis=1)
    print(premium_result.shape)
    premium_result = premium_result.astype('float64')
    result = model.predict(premium_result)

    return "Premium Price : Rs " + str(result[0])

# start the application
app.run(host="0.0.0.0", port=8000, debug=True)
