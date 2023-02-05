import requests
url = "http://127.0.0.1:5000/"

rout = "row/"
id = "182733"

chemin = url + rout + id

reponse = requests.get(chemin)
