import requests
from bs4 import BeautifulSoup
import csv
import time

# Liste des URLs à scraper avec leurs gouvernorats
GOUVERNORATS = {
    "https://www.lilkre.tn/cat-1-1-Appartement_%C3%A0_louer_Ariana": "Ariana",
    "https://www.lilkre.tn/cat-1-3-Appartement_%C3%A0_louer_Ben_Arous#title": "Ben Arous", 
    "https://www.lilkre.tn/cat-1-4-Appartement_%C3%A0_louer_Bizerte#title": "Bizerte",
    "https://www.lilkre.tn/cat-1-13-Appartement_%C3%A0_louer_La_Manouba": "La Manouba",
    "https://www.lilkre.tn/cat-1-15-Appartement_à_louer_Monastir": "Monastir",
    "https://www.lilkre.tn/cat-1-16-Appartement_%C3%A0_louer_Nabeul#title": "Nabeul",
    "https://www.lilkre.tn/cat-1-17-Appartement_%C3%A0_louer_Sfax#title": "Sfax",
    "https://www.lilkre.tn/cat-1-20-Appartement_%C3%A0_louer_Sousse#title": "Sousse",
    "https://www.lilkre.tn/cat-1-23-Appartement_%C3%A0_louer_Tunis": "Tunis"
}


# Classe pour représenter une annonce
class Annonce:
    def __init__(self, prix, pieces, bains, surface, gouvernorat):
        self.prix = prix
        self.pieces = pieces
        self.bains = bains
        self.surface = surface
        self.gouvernorat = gouvernorat
    
    def to_list(self):
        """Convertir l'annonce en liste pour CSV"""
        return [self.prix, self.pieces, self.bains, self.surface, self.gouvernorat]


# Classe principale pour le scraping
class ScraperImmobilier:
    
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.annonces = []
    
    def telecharger_page(self, url):
        """Télécharger le HTML d'une page"""
        response = requests.get(url, headers=self.headers)
        return response.text
    
    def extraire_annonces(self, html, gouvernorat):
        """Extraire toutes les annonces d'une page HTML"""
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("div", class_="card border-0 hover-change-image")
        
        for card in cards:
            annonce = self.extraire_une_annonce(card, gouvernorat)
            if annonce:
                self.annonces.append(annonce)
    
    def extraire_une_annonce(self, card, gouvernorat):
        """Extraire les données d'une seule annonce"""
        card_body = card.find("div", class_="card-body pt-3 px-0 pb-1")
        
        if not card_body:
            return None
        
        # Extraire le prix
        prix = ""
        p_prix = card_body.find("p", class_="fs-17")

        if p_prix:
            texte_prix = p_prix.get_text()
            if "Dt" in texte_prix:
                prix = texte_prix.split("Dt")[0].strip()
            else:
                prix = texte_prix.strip()
   
        # Extraire pièces, bains, surface
        pieces = ""
        bains = ""
        surface = ""
        
        ul = card_body.find("ul", class_="list-inline mb-0")
        if ul:
            items = ul.find_all("li")
            for item in items:
                texte = item.get_text(strip=True)
                
                if "Piéce" in texte or "Pièce" in texte:
                    pieces = texte.split()[0]
                elif "Bain" in texte:
                    bains = texte.split()[0]
                elif "m²" in texte:
                    surface = texte.split()[0]
        
        return Annonce(prix, pieces, bains, surface, gouvernorat)
    
    def scraper_tous_gouvernorats(self):
        """Scraper toutes les pages des gouvernorats"""
        print("Début du scraping...")
        
        for url, gouvernorat in GOUVERNORATS.items():
            print(f"\nScraping de {gouvernorat}...")
            try:
                html = self.telecharger_page(url)
                self.extraire_annonces(html, gouvernorat)
                print(f"{gouvernorat} terminé")
                time.sleep(2)  # Pause pour ne pas surcharger le serveur
            except Exception as e:
                print(f"Erreur pour {gouvernorat}: {e}")
        
        print(f"\nTotal: {len(self.annonces)} annonces trouvées")
    
    def sauvegarder_csv(self, nom_fichier):
        """Enregistrer les annonces dans un fichier CSV"""
        with open(nom_fichier, "w", newline="", encoding="utf-8") as fichier:
            writer = csv.writer(fichier)
            writer.writerow(["Prix", "Pièces", "Bains", "Surface", "Gouvernorat"])
            
            for annonce in self.annonces:
                writer.writerow(annonce.to_list())
        
        print(f"Données enregistrées dans {nom_fichier}")


# Programme principal
if __name__ == "__main__":
    scraper = ScraperImmobilier()
    scraper.scraper_tous_gouvernorats()
    scraper.sauvegarder_csv("annonces_appartements.csv")
    print("\n Scraping terminé avec succès!")