from flask import Flask
import pandas as pd 
import pickle as pk 
import sklearn
import shap

df = pd.read_csv("df_ok_flask_final.csv")

Shap_glob_false =  pd.read_csv("explainer_false_imbl.csv")
Shap_glob_true =  pd.read_csv("explainer_true_imbl.csv")
Shap_ind_true = pd.read_csv("shap_true_imbl.csv")
Shap_ind_false = pd.read_csv("shap_false_imbl.csv")

Proba = pd.read_csv("proba_imb.csv")

model = pk.load(open('RF.pk', 'rb'))


app=Flask(__name__)

def clean_shap(ser):
    """from horrible shap format do fancy stuff"""

    # sep keys from vals
    keys = ser["Unnamed: 0"]
    values = ser["0"]

    # force list[tuple]
    keys = sorted([(i, j) for i, j in keys.items()], key=lambda i: i[0])
    values = sorted([(i, j) for i, j in values.items()], key=lambda i: i[0])

    # extract data
    keys = [i[1] for i in keys]
    values = [i[1] for i in values]

    # create dict with zip
    dd = {k: round(v, 2) for k, v in zip(keys, values)}

    return dd

@app.route('/')
def home():
    return "Bonjour"
    

@app.route('/first')
def first():
    '''return first row
    '''

    ser = df.iloc[0]
    print(ser)

    dd = ser.to_dict()
    print(dd)
    
    return str(dd)


@app.route('/ids')
def ids():
    '''return a list containing all ids'''
    list_ids = Shap_ind_false.SK_ID_CURR.to_list()

    return str(list_ids)

@app.route('/row/<id>')
def row(id):
    '''get specific row
    '''
    try :
        id = int(id)
    except :
        return "Il faut un nombre!"

    ser = df.loc[df["SK_ID_CURR"] == id]
    print(len(ser))
    if len(ser) == 0:
        return "ça marche po!"
    ser = ser.iloc[0]
    print(ser)


    dd = ser.to_dict()
    print(dd)
    
    return str(dd)


@app.route('/predict/<id>')
def predict(id):
    '''return a prediction linked to a given id '''
    try :
        id = int(id)
    except :
        return "Il faut un nombre!"

    ser = df.loc[df["SK_ID_CURR"] == id]
    print(len(ser))
    if len(ser) == 0:
        return "ça marche po!"
   
    print(ser)
    ser.drop(columns = ["SK_ID_CURR"], inplace = True)
  
    prediction = model.predict(ser)
    

    if prediction == 1:
        return "Le crédit est malheureusement refusé"
    else:
        return "Le crédit est accépté !"



@app.route('/shap_neg') 
def shap_global_neg():
    '''envoie un dictionnaire des valeurs shap en fonction des variables'''
    
    list_shap_neg = Shap_glob_false.to_dict()
    list_shap_neg = clean_shap(list_shap_neg)
    #list_shap_neg = list_shap_neg['Unnamed: 0']
    return str(list_shap_neg)

@app.route('/shap_pos') 
def shap_global_pos():
    '''envoie un dictionnaire des valeurs shap en fonction des variables'''

    list_shap_pos = Shap_glob_true.to_dict()
    list_shap_pos = clean_shap(list_shap_pos)
    return str(list_shap_pos)

@app.route('/shap_neg/<id>')
def shap_ind_neg(id):
    '''get specific shap values linked to a given id '''
    try :
        id = int(id)
    except :
        return "Il faut un nombre!"

    sha = Shap_ind_false.loc[Shap_ind_false["SK_ID_CURR"] == id]
    print(len(sha))
    if len(sha) == 0:
        return "ça marche po!"
    sha = sha.iloc[0]
    print(sha)


    d = pd.DataFrame(sha)
    #d = d['Unnamed: 0']
    print(d)
    
    return str(d)

@app.route('/shap_pos/<id>')
def shap_ind_pos(id):
    '''get specific shap values linked to a given id '''
    try :
        id = int(id)
    except :
        return "Il faut un nombre!"

    sha = Shap_ind_true.loc[Shap_ind_true["SK_ID_CURR"] == id]
    print(len(sha))
    if len(sha) == 0:
        return "ça marche po!"
    sha = sha.iloc[0]
    print(sha)


    d = sha.to_dict()
    print(d)
    return str(d)


    #envoie le liste du score en fonction des variables pour un client
@app.route('/score/<id>')
def probab(id):
    '''get specific shap values linked to a given id '''
    try :
        id = int(id)
    except :
        return "Il faut un nombre!"

    pro = Proba.loc[Proba["SK_ID_CURR"] == id]
    print(len(pro))
    if len(pro) == 0:
        return "ça marche po!"
    pro = pro.iloc[0]
    print(pro)


    ddd = pro.to_list()
    print(ddd)
    
    return str(ddd)



if __name__=="__main__":
    #app.run(debug = True, port = 5000, host = "0.0.0.0")
    #app.run(debug = True, port = 5000, host = "localhost")
    app.run(debug = True, port = 5000, host = "127.0.0.1")