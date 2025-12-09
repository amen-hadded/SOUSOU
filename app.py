import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px
from functools import wraps
from datetime import datetime


st.set_page_config(page_title="Prediction Prix Location Tunisie v1", layout="wide")

GOUVERNORATS_TN = {
    'tunis': 'Tunis', 'tunisie': 'Tunis', 'tunisia': 'Tunis',
    'ariana': 'Ariana', 'arianah': 'Ariana', 'ben arous': 'Ben Arous',
    'bizerte': 'Bizerte', 'nabeul': 'Nabeul', 'manouba': 'Manouba',
    'beja': 'Beja', 'jendouba': 'Jendouba', 'kef': 'Kef',
    'kasserine': 'Kasserine', 'sidibouzid': 'Sidi Bouzid',
    'sousse': 'Sousse', 'monastir': 'Monastir', 'mahdia': 'Mahdia',
    'sfax': 'Sfax', 'gabes': 'Gabes', 'medenine': 'Medenine',
    'tataouine': 'Tataouine', 'gafsa': 'Gafsa', 'tozeur': 'Tozeur',
    'kebili': 'Kebili', 'zaghouan': 'Zaghouan', 'siliana': 'Siliana',
    'kairouan': 'Kairouan'
}


# ============== DECORATEUR ==============
def cache_predictions(func):
    """Décorateur pour mettre en cache les prédictions et éviter les recalculs"""
    cache = {}
    
    @wraps(func)
    def wrapper(self, features_array):
        # Créer une clé unique basée sur les features
        cache_key = tuple(features_array.flatten())
        
        # Si déjà calculé, retourner depuis le cache
        if cache_key in cache:
            return cache[cache_key]
        
        # Sinon, calculer et stocker dans le cache
        result = func(self, features_array)
        cache[cache_key] = result
        
        # Afficher stats du cache
        if len(cache) > 1:
            st.caption(f"Cache: {len(cache)} prédictions mémorisées")
        
        return result
    
    # Ajouter une méthode pour vider le cache
    wrapper.clear_cache = lambda: cache.clear()
    wrapper.cache_size = lambda: len(cache)
    
    return wrapper


# ============== HERITAGE ==============
class BaseProcessor:
    """Classe de base pour le traitement des données"""
    def __init__(self):
        self.data = None
    
    def nettoyer_donnees(self, df):
        """Méthode de base pour nettoyer les données"""
        df = df.drop_duplicates()
        df = df.dropna()
        return df
    
    def valider_donnees(self, df):
        """Méthode abstraite à implémenter par les classes filles"""
        raise NotImplementedError("Cette méthode doit être implémentée")


class DataLoader(BaseProcessor):
    """Hérite de BaseProcessor pour charger et traiter les données"""
    def __init__(self, fichier='annonces_appartements.csv'):
        super().__init__()
        self.fichier = fichier
        self.df = None
        self.governorat_mapping = None
        self.governorat_names = None
    
    def valider_donnees(self, df):
        """Implémentation de la validation spécifique aux locations"""
        df = df[(df['Prix'] > 0) & (df['Surface'] > 0) & 
                (df['Pièces'] > 0) & (df['Prix'] < 10000) & 
                (df['Surface'] < 500)]
        return df
    
    def traiter_tout(self):
        try:
            self.df = pd.read_csv(self.fichier)
            self.df = self.nettoyer_donnees(self.df)
            self.df = self.df.dropna(subset=['Prix', 'Surface', 'Pièces', 'Gouvernorat'])
            self.df = self.valider_donnees(self.df)
            
            self.df['prix_par_m2'] = self.df['Prix'] / self.df['Surface']
            self.df['surface_par_piece'] = self.df['Surface'] / self.df['Pièces']
            
            label_encoder = LabelEncoder()
            self.df['Gouvernorat_encoded'] = label_encoder.fit_transform(self.df['Gouvernorat'])
            self.governorat_mapping = {gov: code for gov, code in 
                                      zip(self.df['Gouvernorat'], self.df['Gouvernorat_encoded'])}
            self.governorat_names = list(self.df['Gouvernorat'].unique())
            return True
        except Exception as e:
            st.error(f"Erreur de chargement des donnees: {e}")
            return False


class BaseModele:
    """Classe de base pour les modèles de machine learning"""
    def __init__(self, df):
        self.df = df
        self.model = None
        self.features = []
    
    def preparer_donnees(self):
        """Méthode de base pour préparer les données"""
        raise NotImplementedError("Cette méthode doit être implémentée")
    
    def entrainer(self):
        """Méthode de base pour entraîner le modèle"""
        raise NotImplementedError("Cette méthode doit être implémentée")


class ModeleRegression(BaseModele):
    """Hérite de BaseModele pour la régression linéaire"""
    def __init__(self, df):
        super().__init__(df)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = ['Pièces', 'Bains', 'Surface', 'Gouvernorat_encoded', 'surface_par_piece']
    
    def preparer_donnees(self):
        """Implémentation de la préparation des données"""
        X = self.df[self.features].values
        y = self.df['Prix'].values
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def entrainer_modele(self):
        """Entraîne le modèle"""
        X_train, X_test, y_train, y_test = self.preparer_donnees()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    @cache_predictions
    def predire(self, features_array):
        """Prédit le prix"""
        features_scaled = self.scaler.transform(features_array.reshape(1, -1))
        return self.model.predict(features_scaled)[0]


class ExtracteurTexte:
    """Extrait les caractéristiques depuis le texte en langage naturel"""
    def extraire(self, text):
        text = text.lower().strip()
        features = {'pieces': 3, 'bains': 1, 'surface': 100, 'gouvernorat': 'Tunis', 'detected': False}
        
        # Gouvernorat
        for gov_key, gov_name in GOUVERNORATS_TN.items():
            if gov_key in text:
                features['gouvernorat'] = gov_name
                features['detected'] = True
                break
        
        # Surface
        for pattern in [r'(\d+)\s*m\s*[²2]?', r'(\d+)\s*metres', r'espace\s*(\d+)']:
            match = re.search(pattern, text)
            if match:
                surface = int(match.group(1))
                if 20 <= surface <= 500:
                    features['surface'] = surface
                    features['detected'] = True
                    break
        
        # Pieces
        for pattern in [r's\s*[+\-]\s*(\d+)', r'(\d+)\s*pi[èe]ces', r'(\d+)\s*chambres', r't(\d+)']:
            match = re.search(pattern, text)
            if match:
                pieces = int(match.group(1))
                if 1 <= pieces <= 10:
                    features['pieces'] = pieces
                    features['detected'] = True
                    features['bains'] = 1 if pieces <= 2 else (2 if pieces <= 4 else min(pieces // 2, 5))
                    break
        
        # Bains
        for pattern in [r'(\d+)\s*bains', r'(\d+)\s*sdb']:
            match = re.search(pattern, text)
            if match:
                bains = int(match.group(1))
                if 1 <= bains <= 5:
                    features['bains'] = bains
                    features['detected'] = True
                    break
        
        return features


class InterfaceStreamlit:
    def __init__(self):
        self.data_loader = DataLoader()
        self.modele = None
        self.extracteur = ExtracteurTexte()
    
    def initialiser(self):
        with st.spinner('Chargement...'):
            if not self.data_loader.traiter_tout():
                st.stop()
            self.modele = ModeleRegression(self.data_loader.df)
            self.modele.entrainer_modele()
    
    def afficher_prix(self, pieces, bains, surface, gouvernorat, titre="Prix estime"):
        """Affiche les resultats de prediction"""
        surface_par_piece = surface / pieces
        gouvernorat_encoded = self.data_loader.governorat_mapping.get(gouvernorat, 0)
        
        features_array = np.array([pieces, bains, surface, gouvernorat_encoded, surface_par_piece])
        prix_pred = self.modele.predire(features_array)
        
        st.markdown("---")
        st.subheader(titre)
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Par mois", f"{prix_pred:,.0f} TND")
        with col_y:
            st.metric("Par jour", f"{prix_pred/30:,.0f} TND")
        with col_z:
            st.metric("Au m²", f"{prix_pred/surface:,.0f} TND/m²")
        
        return prix_pred
    
    def afficher_suggestions(self, gouvernorat, pieces, surface):
        """Affiche les suggestions similaires"""
        st.markdown("---")
        st.subheader("Suggestions similaires")
        
        df = self.data_loader.df
        similar = df[(df['Gouvernorat'] == gouvernorat) &
                    (df['Pièces'].between(pieces-1, pieces+1)) &
                    (df['Surface'].between(surface-20, surface+20))]
        
        if len(similar) > 0:
            similar = similar.nsmallest(5, 'Prix')
            for idx, row in similar.iterrows():
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.write(f"**{row['Pièces']} pieces, {row.get('Bains', 1)} bain(s), {row['Surface']} m²**")
                with cols[1]:
                    st.write(f"{row['Gouvernorat']}")
                with cols[2]:
                    st.write(f"{row['Prix']:,.0f} TND")
                with cols[3]:
                    st.write(f"{row['prix_par_m2']:.0f} TND/m²")
        else:
            st.info("Aucune annonce similaire trouvee")
    
    def executer(self):
        st.title("Recherche Intelligente - Prix Location Tunisie")
        st.markdown("Decrivez ce que vous cherchez en langage naturel")
        st.markdown("Exemples : 'maison a ariana s+2 100m²' ou 'appartement 3 pieces tunis'")
        
        # Recherche
        recherche = st.text_input("Decrivez votre recherche :", 
                                  placeholder="Ex: appartement a ariana s+2 avec 100m²")
        
        if recherche:
            extracted = self.extracteur.extraire(recherche)
            
            if extracted['detected']:
                st.markdown("---")
                st.subheader("Ce que j'ai compris :")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Gouvernorat", extracted['gouvernorat'])
                with col_b:
                    st.metric("Pieces", extracted['pieces'])
                with col_c:
                    st.metric("Bains", extracted['bains'])
                with col_d:
                    st.metric("Surface", f"{extracted['surface']} m²")
                
                # Afficher le prix
                self.afficher_prix(extracted['pieces'], extracted['bains'], 
                                 extracted['surface'], extracted['gouvernorat'])
                
                # Suggestions
                self.afficher_suggestions(extracted['gouvernorat'], 
                                        extracted['pieces'], extracted['surface'])
            else:
                st.warning("Je n'ai pas bien compris. Essayez : 'ariana s+2 100m²' ou '3 pieces tunis 120m'")
        
        # Ajustements manuels
        st.markdown("---")
        st.subheader("Ajustements manuels")
        
        with st.form("manual_adjust"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pieces_adj = st.slider("Pieces", 1, 5, 1)
            with col2:
                bains_adj = st.slider("Bains", 1, 3, 1)
            with col3:
                surface_adj = st.slider("Surface (m²)", 20, 300, 10)
            with col4:
                gouvernorat_adj = st.selectbox("Gouvernorat", sorted(self.data_loader.governorat_names))
            
            ajuster = st.form_submit_button("Recalculer", use_container_width=True)
        
        if ajuster:
            self.afficher_prix(pieces_adj, bains_adj, surface_adj, gouvernorat_adj, "Resultat ajuste")
            self.afficher_suggestions(gouvernorat_adj, pieces_adj, surface_adj)
        
        # Exploration
        st.markdown("---")
        st.subheader("Exploration des donnees")
        df = self.data_loader.df
        prix_par_gouvernorat = df.groupby('Gouvernorat')['Prix'].mean().sort_values(ascending=False)
        fig = px.bar(y=prix_par_gouvernorat.index, x=prix_par_gouvernorat.values,
                    orientation='h', title="Prix moyen par gouvernorat",
                    labels={'x': 'Prix moyen (TND)', 'y': 'Gouvernorat'})
        st.plotly_chart(fig, use_container_width=True)


def main():
    app = InterfaceStreamlit()
    app.initialiser()
    app.executer()

if __name__ == "__main__":
    main()
