import streamlit as st
import requests
import pandas as pd
from PIL import Image
#import plotly.express as px
import numpy as np

base = "http://ec2-18-204-11-80.compute-1.amazonaws.com:5000"
#base = 'http://127.0.0.1:5000'


st.write('''# Bienvenue,
cette application prédit l'obtention d'un prêt bancaire
''')

def user_input():
    '''Récupérer l'ensemble des ids de la BBD'''

    r = requests.get(base + '/ids')
    assert int(r.status_code) == 200

    ids = eval(r.text)
    return ids

def user_input():
    '''Récupérer l'ensemble des ids de la BBD'''

    r = requests.get(base + '/ids')
    assert int(r.status_code) == 200

    ids = eval(r.text)
    return ids

def get_informations(ids):
    """get informations of a client based on his id"""

    r = requests.get(base + "/row/" + str(ids) + "")
    assert int(r.status_code) == 200

    info = eval(r.text)
    info = pd.DataFrame.from_dict(info, orient='index')
    infos = st.table(info)

def predict(ids):
    """predict and id"""

    r = requests.get(base + "/predict/" + str(ids))
    assert int(r.status_code) == 200

    ans = r.text

    assert ans in ["Le crédit est malheureusement refusé", "Le crédit est accépté !"]
    return ans

def get_score(ids):
    """get informations of a client based on his id"""

    r = requests.get(base + "/score/" + str(ids) + "")
    assert int(r.status_code) == 200

    info = eval(r.text)
    #info = pd.DataFrame.from_dict(info, orient='index')
    print(info[0])
    infos = st.write(info[0])
    

# selection
list_ids = user_input()
option2 = st.selectbox("Qui est le client?", list_ids)
st.write("Vous avez sélectionné:", option2)

print(option2)

st.write("Les informations du client sont :")
# Print information of a client
info_client = get_informations(option2)
#st.write("Vos informations sont:", infos)

# pred
#if isinstance(option2, int):
    #ans = predict(option2)
    #ans = "Le crédit est accépté !" if ans else "Le crédit est malheureusement refusé"

ans = predict(option2)
st.write(f"La réponse est : {ans}  Les explications de cette décision sont données par les deux graphiques ci dessous.")


def shap_glob():
    '''Récupérer l'ensemble des shap_values des variables'''

    r = requests.get(base + '/shap_pos')
    assert int(r.status_code) == 200

    ids = eval(r.text)
    print(ids)
    return ids

def get_shap_ind(ids):
    """get informations of a client based on his id"""

    r = requests.get(base + "/shap_pos/" + str(ids) + "")
    assert int(r.status_code) == 200

    info = eval(r.text)
    print(info)
    return info
    #info = pd.DataFrame.from_dict(info, orient='index')
    #info = info.drop("SK_ID_CURR")
    #infos = st.table(info)

#Shap global:
st.write("Ce premier graphique montre l'importance de chaque information dans l'établissement du modèle d'optention du crédit")
shap = shap_glob()
shap = pd.DataFrame([pd.Series(shap)])
shap = shap.drop(columns = "SK_ID_CURR")
shap = pd.melt(shap)
st.bar_chart(data=shap, x="variable", y="value")

#Shap_ind
st.write("Ce deuxième graphique montre l'importance des informations du client pour la réponse de la banque au client")
shap_ind = get_shap_ind(option2)
shap_ind = pd.DataFrame([pd.Series(shap_ind)])
shap_ind = shap_ind.drop(columns = "SK_ID_CURR")
shap_ind = shap_ind.drop(columns = "Unnamed: 0")

shap_ind = pd.melt(shap_ind)
st.bar_chart(data=shap_ind, x="variable", y="value")

# Score
st.write("Le score  du client est :")
# Print information of a client
score_ind = get_score(option2)
#st.write("Vos informations sont:", infos)

