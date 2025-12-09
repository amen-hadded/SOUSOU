import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Location Tunisie",
    page_icon="🏠",
    layout="wide"
)

# Dictionnaire des gouvernorats tunisiens avec variations d'écriture
GOUVERNORATS_TN = {
    'tunis': 'Tunis',
    'tunisie': 'Tunis',
    'tunisia': 'Tunis',
    'ariana': 'Ariana',
    'arianah': 'Ariana',
    'ben arous': 'Ben Arous',
    'bizerte': 'Bizerte',
    'bizerteh': 'Bizerte',
    'nabeul': 'Nabeul',
    'manouba': 'Manouba',
    'manoubah': 'Manouba',
    'beja': 'Béja',
    'bejaa': 'Béja',
    'jendouba': 'Jendouba',
    'kef': 'Kef',
    'kasserine': 'Kasserine',
    'sidibouzid': 'Sidi Bouzid',
    'sousse': 'Sousse',
    'monastir': 'Monastir',
    'mahdia': 'Mahdia',
    'sfax': 'Sfax',
    'gabes': 'Gabès',
    'gabesah': 'Gabès',
    'medenine': 'Médenine',
    'tataouine': 'Tataouine',
    'gafsa': 'Gafsa',
    'tozeur': 'Tozeur',
    'kebili': 'Kébili',
    'zaghouan': 'Zaghouan',
    'siliana': 'Siliana',
    'kairouan': 'Kairouan'
}

# Cache les données et le modèle
@st.cache_data
def load_and_process_data():
    """Charge et traite les données"""
    try:
        df = pd.read_csv('annonces_appartements.csv')
        
        # Nettoyage
        df = df.drop_duplicates()
        df = df.dropna(subset=['Prix', 'Surface', 'Pièces', 'Gouvernorat'])
        
        # Filtrer les valeurs aberrantes
        df = df[
            (df['Prix'] > 0) & 
            (df['Surface'] > 0) & 
            (df['Pièces'] > 0) &
            (df['Prix'] < 10000) &
            (df['Surface'] < 500)
        ]
        
        # Colonnes supplémentaires
        df['prix_par_m2'] = df['Prix'] / df['Surface']
        df['surface_par_piece'] = df['Surface'] / df['Pièces']
        
        # Encoder le gouvernorat
        label_encoder = LabelEncoder()
        df['Gouvernorat_encoded'] = label_encoder.fit_transform(df['Gouvernorat'])
        
        governorat_mapping = {gov: code for gov, code in 
                            zip(df['Gouvernorat'], df['Gouvernorat_encoded'])}
        governorat_names = list(df['Gouvernorat'].unique())
        
        return df, governorat_mapping, governorat_names, label_encoder
        
    except FileNotFoundError:
        st.error("Fichier 'annonces_appartements.csv' non trouvé !")
        st.stop()
    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        st.stop()

@st.cache_resource
def train_model(df):
    """Entraîne le modèle"""
    features = ['Pièces', 'Bains', 'Surface', 'Gouvernorat_encoded', 'surface_par_piece']
    
    X = df[features].values
    y = df['Prix'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Prédictions et métriques
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_pred_train),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
        },
        'test': {
            'r2': r2_score(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
    }
    
    return model, scaler, features, metrics

def extract_features_from_text(text):
    """
    Extrait les caractéristiques d'une recherche en langage naturel
    Exemples :
    - "je veux une maison à ariana s+2 avec espace 100m"
    - "ariana s+2 100"
    - "100 s+2 ariana"
    - "appartement 3 pièces 2 bains 120m² tunis"
    """
    text = text.lower().strip()
    
    # Initialisation des valeurs par défaut
    features = {
        'pieces': 3,      # Valeur par défaut
        'bains': 1,       # Valeur par défaut
        'surface': 100,   # Valeur par défaut
        'gouvernorat': 'Tunis',  # Valeur par défaut
        'detected': False
    }
    
    # Détection du gouvernorat
    gouvernorat_trouve = None
    for gov_key, gov_name in GOUVERNORATS_TN.items():
        if gov_key in text:
            gouvernorat_trouve = gov_name
            break
    
    if gouvernorat_trouve:
        features['gouvernorat'] = gouvernorat_trouve
        features['detected'] = True
    
    # Détection de la surface (nombre suivi de m, m², m2, mètres, etc.)
    surface_patterns = [
        r'(\d+)\s*m\s*[²2]?',
        r'(\d+)\s*mètres',
        r'(\d+)\s*m2',
        r'espace\s*(\d+)',
        r'surface\s*(\d+)',
        r'(\d+)\s*m\b'
    ]
    
    for pattern in surface_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                surface = int(match.group(1))
                if 20 <= surface <= 500:  # Plage réaliste
                    features['surface'] = surface
                    features['detected'] = True
                    break
            except:
                pass
    
    # Détection du nombre de pièces (s+2, s+3, s+4, etc.)
    pieces_patterns = [
        r's\s*[+\-]\s*(\d+)',          # s+2, s+3
        r'(\d+)\s*pi[èe]ces',          # 2 pièces, 3 pieces
        r'(\d+)\s*chambres',           # 2 chambres
        r'(\d+)\s*salles',             # 2 salles
        r'appart\s*(\d+)',             # appart 2
        r'(\d+)\s*piece',              # 2 piece
        r't(\d+)'                      # t2, t3, t4
    ]
    
    for pattern in pieces_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                pieces = int(match.group(1))
                if 1 <= pieces <= 10:  # Plage réaliste
                    features['pieces'] = pieces
                    features['detected'] = True
                    
                    # Si on a le nombre de pièces, on peut estimer les bains
                    if pieces <= 2:
                        features['bains'] = 1
                    elif pieces <= 4:
                        features['bains'] = 2
                    else:
                        features['bains'] = min(pieces // 2, 5)
                    break
            except:
                pass
    
    # Détection spécifique du nombre de bains
    bains_patterns = [
        r'(\d+)\s*bains',
        r'(\d+)\s*sdb',
        r'(\d+)\s*salle[s]?\s*de\s*bain'
    ]
    
    for pattern in bains_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                bains = int(match.group(1))
                if 1 <= bains <= 5:
                    features['bains'] = bains
                    features['detected'] = True
                    break
            except:
                pass
    
    # Recherche de nombres simples pour la surface et pièces
    if not features['detected']:
        numbers = re.findall(r'\b(\d{2,3})\b', text)
        numbers = [int(n) for n in numbers if n.isdigit()]
        
        if len(numbers) >= 2:
            # Le plus grand nombre est probablement la surface
            surface_candidate = max(numbers)
            pieces_candidate = min(numbers)
            
            if 20 <= surface_candidate <= 500:
                features['surface'] = surface_candidate
                features['detected'] = True
            
            if 1 <= pieces_candidate <= 10:
                features['pieces'] = pieces_candidate
                features['detected'] = True
                if pieces_candidate <= 2:
                    features['bains'] = 1
                elif pieces_candidate <= 4:
                    features['bains'] = 2
                else:
                    features['bains'] = min(pieces_candidate // 2, 5)
    
    return features

def main():
    # En-tête
    st.title("🏠 Recherche Intelligente - Prix Location Tunisie")
    st.markdown("""
    **Décrivez ce que vous cherchez en langage naturel** : 
    *Exemples : "maison a ariana s+2 100m²" ou "appartement 3 pièces 2 bains 120m² tunis"*
    """)
    
    # Chargement des données
    with st.spinner('Chargement des données...'):
        df, governorat_mapping, governorat_names, label_encoder = load_and_process_data()
    
    # Entraînement du modèle
    with st.spinner('Entraînement du modèle...'):
        model, scaler, features, metrics = train_model(df)
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Barre de recherche en langage naturel
        st.subheader("🔍 Recherche en langage naturel")
        recherche = st.text_input(
            "Décrivez votre recherche :",
            placeholder="Ex: 'je veux un appartement à ariana s+2 avec 100m²'",
            help="Vous pouvez écrire comme vous parlez !"
        )
        
        # Boutons d'exemples rapides
        st.markdown("**Exemples rapides :**")
        examples_cols = st.columns(4)
        with examples_cols[0]:
            if st.button("Ariana s+2 100m²", use_container_width=True):
                st.session_state.recherche = "ariana s+2 100m²"
                st.rerun()
        with examples_cols[1]:
            if st.button("Tunis 3 pièces 120m", use_container_width=True):
                st.session_state.recherche = "tunis 3 pièces 120m"
                st.rerun()
        with examples_cols[2]:
            if st.button("Sousse t4 150m²", use_container_width=True):
                st.session_state.recherche = "sousse t4 150m²"
                st.rerun()
        with examples_cols[3]:
            if st.button("Bizerte 2 chambres 90m", use_container_width=True):
                st.session_state.recherche = "bizerte 2 chambres 90m"
                st.rerun()
        
        if recherche:
            # Extraction des caractéristiques
            extracted = extract_features_from_text(recherche)
            
            # Affichage de ce qui a été détecté
            st.markdown("---")
            st.subheader("📋 Ce que j'ai compris :")
            
            if extracted['detected']:
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("📍 Gouvernorat", extracted['gouvernorat'])
                with col_b:
                    st.metric("🛏️ Pièces", extracted['pieces'])
                with col_c:
                    st.metric("🚿 Bains", extracted['bains'])
                with col_d:
                    st.metric("📏 Surface", f"{extracted['surface']} m²")
                
                # Calcul de la surface par pièce
                surface_par_piece = extracted['surface'] / extracted['pieces']
                
                # Préparation pour la prédiction
                try:
                    gouvernorat_encoded = governorat_mapping.get(extracted['gouvernorat'], 
                                                               governorat_mapping.get('Tunis', 0))
                    
                    features_array = np.array([
                        extracted['pieces'],
                        extracted['bains'],
                        extracted['surface'],
                        gouvernorat_encoded,
                        surface_par_piece
                    ])
                    
                    features_scaled = scaler.transform(features_array.reshape(1, -1))
                    prix_pred = model.predict(features_scaled)[0]
                    
                    # Affichage du résultat
                    st.markdown("---")
                    st.subheader("💰 Prix estimé")
                    
                    col_x, col_y, col_z = st.columns(3)
                    with col_x:
                        st.metric("Par mois", f"{prix_pred:,.0f} TND", 
                                 help="Prix mensuel estimé")
                    with col_y:
                        st.metric("Par jour", f"{prix_pred/30:,.0f} TND",
                                 help="Prix journalier approximatif")
                    with col_z:
                        prix_m2 = prix_pred / extracted['surface']
                        st.metric("Au m²", f"{prix_m2:,.0f} TND/m²",
                                 help="Prix au mètre carré")
                    
                    # Graphique
                    fig = px.bar(
                        x=["Prix estimé"],
                        y=[prix_pred],
                        title="Estimation du prix de location",
                        labels={'x': '', 'y': 'Prix (TND)'},
                        text=[f"{prix_pred:,.0f} TND"],
                        color_discrete_sequence=['#2E86AB']
                    )
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur dans la prédiction : {str(e)}")
                    
                # Suggestions d'alternatives
                st.markdown("---")
                st.subheader("💡 Suggestions similaires")
                
                # Recherche d'annonces similaires
                similar = df[
                    (df['Gouvernorat'] == extracted['gouvernorat']) &
                    (df['Pièces'].between(extracted['pieces']-1, extracted['pieces']+1)) &
                    (df['Surface'].between(extracted['surface']-20, extracted['surface']+20))
                ]
                
                if len(similar) > 0:
                    similar = similar.nsmallest(5, 'Prix')
                    
                    for idx, row in similar.iterrows():
                        with st.container():
                            cols = st.columns([3, 1, 1, 1])
                            with cols[0]:
                                st.write(f"**{row['Pièces']} pièces, {row.get('Bains', 1)} bain(s), {row['Surface']} m²**")
                            with cols[1]:
                                st.write(f"📍 {row['Gouvernorat']}")
                            with cols[2]:
                                st.write(f"💰 {row['Prix']:,.0f} TND")
                            with cols[3]:
                                st.write(f"📐 {row['prix_par_m2']:.0f} TND/m²")
                            st.markdown("---")
                else:
                    st.info("Aucune annonce similaire trouvée dans notre base de données.")
            else:
                st.warning("""
                Je n'ai pas bien compris votre recherche. Essayez de formuler comme :
                - "ariana s+2 100m²"
                - "3 pièces tunis 120m"
                - "maison bizerte 4 chambres 150m²"
                """)
    
    with col2:
        # Panneau d'information et ajustement manuel
        st.subheader("⚙️ Ajustements manuels")
        
        with st.form("manual_adjust"):
            if 'extracted' in locals() and extracted['detected']:
                pieces_adj = st.slider("Pièces", 1, 10, extracted['pieces'])
                bains_adj = st.slider("Bains", 1, 5, extracted['bains'])
                surface_adj = st.slider("Surface (m²)", 20, 300, extracted['surface'])
                
                gouvernorats_liste = sorted(governorat_names)
                index_gouv = gouvernorats_liste.index(extracted['gouvernorat']) if extracted['gouvernorat'] in gouvernorats_liste else 0
                gouvernorat_adj = st.selectbox("Gouvernorat", gouvernorats_liste, index=index_gouv)
            else:
                pieces_adj = st.slider("Pièces", 1, 10, 3)
                bains_adj = st.slider("Bains", 1, 5, 1)
                surface_adj = st.slider("Surface (m²)", 20, 300, 100)
                gouvernorat_adj = st.selectbox("Gouvernorat", sorted(governorat_names))
            
            ajuster = st.form_submit_button("Recalculer avec ces valeurs")
        
        if ajuster:
            surface_par_piece_adj = surface_adj / pieces_adj
            gouvernorat_encoded_adj = governorat_mapping[gouvernorat_adj]
            
            features_array_adj = np.array([
                pieces_adj,
                bains_adj,
                surface_adj,
                gouvernorat_encoded_adj,
                surface_par_piece_adj
            ])
            
            features_scaled_adj = scaler.transform(features_array_adj.reshape(1, -1))
            prix_pred_adj = model.predict(features_scaled_adj)[0]
            
            st.success(f"**Prix ajusté : {prix_pred_adj:,.0f} TND/mois**")
        
        # Statistiques
        st.markdown("---")
        st.subheader("📊 Statistiques du marché")
        
        avg_price = df['Prix'].mean()
        avg_size = df['Surface'].mean()
        avg_price_m2 = df['prix_par_m2'].mean()
        
        st.metric("💰 Prix moyen", f"{avg_price:,.0f} TND")
        st.metric("📐 Surface moyenne", f"{avg_size:.0f} m²")
        st.metric("📈 Prix moyen/m²", f"{avg_price_m2:.0f} TND")
        
        # Top 5 gouvernorats les plus chers
        top_gouvernorats = df.groupby('Gouvernorat')['prix_par_m2'].mean().nlargest(5)
        st.markdown("**Top 5 gouvernorats (prix/m²) :**")
        for gov, price in top_gouvernorats.items():
            st.write(f"- {gov}: {price:.0f} TND/m²")
    
    # Onglets inférieurs
    tab1, tab2 = st.tabs(["📈 Performance du modèle", "📊 Exploration des données"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{metrics['test']['r2']:.3f}")
            st.metric("MAE", f"{metrics['test']['mae']:.0f} TND")
            st.metric("RMSE", f"{metrics['test']['rmse']:.0f} TND")
        
        with col2:
            # Importance des features
            importance = pd.DataFrame({
                'Caractéristique': features,
                'Importance': abs(model.coef_)
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance,
                x='Importance',
                y='Caractéristique',
                orientation='h',
                title="Importance des caractéristiques",
                color='Importance',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Distribution des prix
        fig = px.histogram(
            df,
            x='Prix',
            nbins=50,
            title="Distribution des prix",
            labels={'Prix': 'Prix (TND)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prix par gouvernorat
        prix_par_gouvernorat = df.groupby('Gouvernorat')['Prix'].mean().sort_values(ascending=True)
        fig = px.bar(
            y=prix_par_gouvernorat.index,
            x=prix_par_gouvernorat.values,
            orientation='h',
            title="Prix moyen par gouvernorat",
            labels={'x': 'Prix moyen (TND)', 'y': 'Gouvernorat'}
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()