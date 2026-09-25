"""
Application Streamlit — Prédiction de la souscription d'un dépôt à terme d'une client(term deposit)cc
Conversion directe de l'application Gradio d'origine.

Lancement en local :  streamlit run app_bank.py
"""

import numpy as np 
import pandas as pd 
import joblib as jb 
import streamlit as st 
from xgboost import XGBClassifier


# Configuration de la page
st.set_page_config(
    page_title="Prédiction de la souscription d'un dépôt à terme Model xgboot",
    page_icon="🏦",
    layout="centered",
)

DESCRIPTION = (
    "Ce jeu de données contient des informations issues de campagnes de marketing téléphonique réalisées par une banque portugaise."
    "L'objectif est de prédire si un client souscrira à un dépôt à terme (y)."
    "Model xgboot"

)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (Marque, Quartier, Transmission)
    uniques = jb.load("uniques.joblib")     # valeurs uniques
    scaler = jb.load("scaler.joblib")       # normaliseur
    xgb = XGBClassifier()
    xgb.load_model("xgb_model.json")
    return encoders, uniques, scaler, xgb


encoders, uniques, scaler, xgb = load_artifacts()
clasnames = uniques[9]  # noms des classes



# Fonction de prédiction simple

def Pred_func(age,	job,	marital,	education,	housing,	loan,	contact,	month,	day_of_week,	duration,	campaign,	pdays,	previous,	poutcome):

  # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission
   job = encoders[0].transform([job])[0]
   marital= encoders[1].transform([marital])[0]
   education= encoders[2].transform([education])[0]
   housing= encoders[3].transform([housing])[0]
   loan = encoders[4].transform([loan])[0]
   contact = encoders[5].transform([contact])[0]
   month= encoders[6].transform([month])[0]
   day_of_week = encoders[7].transform([day_of_week])[0]
   poutcome= encoders[8].transform([poutcome])[0]

  # vecteur des valeurs numériques
   x_new = np.array([age,	job,	marital,	education,	housing,	loan,	contact,	month,	day_of_week,	duration,	campaign,	pdays,	previous,	poutcome])
   x_new = x_new.reshape(1,-1) # convert en un 2D array
  # Normaliser les données
   x_new = scaler.transform(x_new)
  # Prédire
   y_pred = xgb.predict(x_new)
   return clasnames[y_pred[0]]



# Fonction de prédiction multiple

# Fonction de prédiction multiple
def Pred_func_csv(file):
  # Lire le fichier csv
  df = pd.read_csv(file)
  predictions = []
  # Boucle sur les lignes du dataframe
  for row in df.iloc[:, :].values:
    # prédiction simple
    y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5],row[6], row[7], row[8], row[9], row[10], row[11],row[12], row[13])
    predictions.append(y_pred)

  df['y'] = predictions
  df.to_csv('predictions.csv', index = False)
  return 'predictions.csv'



# Interface

st.title("🏦 Prédiction de la souscription d'un dépôt à terme")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire  la souscription d'un dépôt à terme avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", value=0, step=1, format="%d")
            job = st.selectbox("Job", options=list(uniques[0]))
            marital= st.selectbox("Marital", options=list(uniques[1]))
            education= st.selectbox("Education", options=list(uniques[2]))
            housing= st.selectbox("Housing", options=list(uniques[3]))
            loan= st.selectbox("Loan", options=list(uniques[4]))
            contact = st.selectbox("Contact", options=list(uniques[5]))
        with col2:
            month = st.selectbox("Month", options=list(uniques[6]))
            day_of_week= st.selectbox("Day_of_week", options=list(uniques[7]))
            duration = st.number_input("Duration", value=0, step=1, format="%d")
            campaign = st.number_input("Campaign", value=0, step=1, format="%d")
            pdays = st.number_input("Pdays", value=0, step=1, format="%d")
            previous = st.number_input("Previous", value=0, step=1, format="%d")
            poutcome = st.selectbox("Poutcome", options=list(uniques[8]))
            

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(age, job, marital, education, housing,	loan, contact, month, day_of_week, duration, campaign,	pdays,	previous,	poutcome)  # type: ignore
            st.success(f"**Souscription dépôt à terme :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la souscription d'un dépôt à terme avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "age,	job,	marital,	education,	housing,	loan,	contact,	month,	day_of_week,	"
        "duration,	campaign,	pdays,	previous,	poutcome."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
