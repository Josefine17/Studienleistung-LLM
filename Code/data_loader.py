# data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
import re
from html import unescape

class IMDBDataLoader:
    """
    Klasse zum Laden und Vorbereiten des IMDB Datasets für Sentiment-Analyse
    
    Diese Klasse bietet Methoden zum:
    - Laden des IMDB Datasets
    - Bereinigen von HTML-Tags und Sonderzeichen
    - Erstellen von Samples
    - Anzeigen von Statistiken
    """
    
    def __init__(self, data_path='data/IMDB Dataset.csv'):
        """
        Initialisiert den Data Loader
        
        Args:
            data_path: Pfad zur IMDB Dataset CSV-Datei
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self):
        """
        Lädt das IMDB Dataset von der angegebenen CSV-Datei
        
        Returns:
            pandas.DataFrame: Das geladene Dataset
            
        Raises:
            FileNotFoundError: Wenn die Datei nicht existiert
        """
        # Prüfen ob Datei existiert
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"❌ Dataset nicht gefunden unter: {self.data_path}\n\n"
                f"Bitte laden Sie es herunter von:\n"
                f"https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews\n"
                f"und speichern Sie es unter: {self.data_path}"
            )
        
        print(f"📂 Lade Dataset von: {self.data_path}")
        
        # CSV-Datei einlesen
        self.df = pd.read_csv(self.data_path)
        
        # Erfolgsmeldung mit Statistiken
        print(f"✅ Dataset erfolgreich geladen!")
        print(f"   Gesamtanzahl: {len(self.df)} Reviews")
        print(f"   Positive: {(self.df['sentiment'] == 'positive').sum()}")
        print(f"   Negative: {(self.df['sentiment'] == 'negative').sum()}")
        
        return self.df
    
    def clean_text(self, text):
        """
        Bereinigt einen einzelnen Text von HTML-Tags und Sonderzeichen
        
        IMDB Reviews enthalten oft HTML-Tags wie <br />, die entfernt werden müssen.
        
        Args:
            text: Der zu bereinigende Text
            
        Returns:
            str: Bereinigter Text
        """
        # HTML entities dekodieren (z.B. &amp; -> &)
        text = unescape(text)
        
        # <br /> Tags durch Leerzeichen ersetzen
        text = re.sub(r'<br\s*/?>', ' ', text)
        
        # Alle anderen HTML-Tags entfernen
        text = re.sub(r'<[^>]+>', '', text)
        
        # Mehrfache Leerzeichen durch ein Leerzeichen ersetzen
        text = re.sub(r'\s+', ' ', text)
        
        # Führende und nachfolgende Leerzeichen entfernen
        text = text.strip()
        
        return text
    
    def preprocess(self):
        """
        Bereinigt alle Reviews im Dataset
        
        Erstellt eine neue Spalte 'review_cleaned' mit bereinigten Texten
        und eine Spalte 'review_length' mit der Zeichenanzahl.
        
        Returns:
            pandas.DataFrame: Dataset mit zusätzlichen Spalten
        """
        if self.df is None:
            raise ValueError("❌ Bitte zuerst load_data() aufrufen!")
        
        print("\n🧹 Bereinige Reviews...")
        
        # Bereinigung auf alle Reviews anwenden
        self.df['review_cleaned'] = self.df['review'].apply(self.clean_text)
        
        # Länge berechnen
        self.df['review_length'] = self.df['review_cleaned'].str.len()
        
        print("✅ Reviews bereinigt!")
        print(f"   Neue Spalten: 'review_cleaned', 'review_length'")
        
        return self.df
    
    def get_sample(self, n=200, stratified=True, random_state=42):
        """
        Erstellt ein Sample-Dataset aus dem Hauptdataset
        
        Args:
            n: Gesamtanzahl der gewünschten Samples
            stratified: Wenn True, wird gleiche Anzahl pro Sentiment genommen
            random_state: Zufallsseed für Reproduzierbarkeit
            
        Returns:
            pandas.DataFrame: Sample-Dataset
        """
        if self.df is None:
            raise ValueError("❌ Bitte zuerst load_data() aufrufen!")
        
        print(f"\n📊 Erstelle Sample mit {n} Reviews...")
        
        if stratified:
            # Stratifiziertes Sampling: Gleiche Anzahl pro Kategorie
            # Damit haben wir eine ausgewogene Verteilung
            sample = self.df.groupby('sentiment', group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), n // 2),  # n/2 pro Sentiment
                    random_state=random_state
                )
            )
            print(f"   Stratifiziert: Ja (gleiche Anzahl pro Sentiment)")
        else:
            # Einfaches zufälliges Sampling
            sample = self.df.sample(n=n, random_state=random_state)
            print(f"   Stratifiziert: Nein (zufällig)")
        
        # Statistiken anzeigen
        print(f"✅ Sample erstellt!")
        print(f"   Gesamt: {len(sample)} Reviews")
        print(f"   Positive: {(sample['sentiment'] == 'positive').sum()}")
        print(f"   Negative: {(sample['sentiment'] == 'negative').sum()}")
        
        return sample
    
    def get_statistics(self):
        """
        Zeigt detaillierte Statistiken über das Dataset
        """
        if self.df is None:
            raise ValueError("❌ Bitte zuerst load_data() aufrufen!")
        
        print("\n" + "="*70)
        print(" "*25 + "DATASET STATISTIKEN")
        print("="*70)
        
        # Grundlegende Informationen
        print(f"\n📈 Grundinformationen:")
        print(f"   Gesamtanzahl Reviews: {len(self.df):,}")
        print(f"   Spalten: {', '.join(self.df.columns)}")
        
        # Sentiment-Verteilung
        print(f"\n💭 Sentiment-Verteilung:")
        sentiment_counts = self.df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        # Review-Längen (falls bereits vorverarbeitet)
        if 'review_length' in self.df.columns:
            print(f"\n📏 Review-Länge (Zeichen):")
            print(f"   Durchschnitt: {self.df['review_length'].mean():.0f}")
            print(f"   Median: {self.df['review_length'].median():.0f}")
            print(f"   Min: {self.df['review_length'].min()}")
            print(f"   Max: {self.df['review_length'].max()}")
            
            # Quartile
            q25 = self.df['review_length'].quantile(0.25)
            q75 = self.df['review_length'].quantile(0.75)
            print(f"   25% Quartil: {q25:.0f}")
            print(f"   75% Quartil: {q75:.0f}")
        
        # Beispiel-Reviews
        print(f"\n📝 Beispiel-Review (Positive):")
        pos_example = self.df[self.df['sentiment'] == 'positive'].iloc[0]
        review_text = pos_example['review_cleaned'] if 'review_cleaned' in self.df.columns else pos_example['review']
        print(f"   {review_text[:250]}...")
        
        print(f"\n📝 Beispiel-Review (Negative):")
        neg_example = self.df[self.df['sentiment'] == 'negative'].iloc[0]
        review_text = neg_example['review_cleaned'] if 'review_cleaned' in self.df.columns else neg_example['review']
        print(f"   {review_text[:250]}...")
        
        print("\n" + "="*70)
    
    def save_processed(self, output_path='data/imdb_processed.csv'):
        """
        Speichert das vorverarbeitete Dataset
        
        Args:
            output_path: Pfad, unter dem das Dataset gespeichert werden soll
        """
        if self.df is None:
            raise ValueError("❌ Bitte zuerst load_data() aufrufen!")
        
        output_path = Path(output_path)
        
        # Verzeichnis erstellen, falls es nicht existiert
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Als CSV speichern
        self.df.to_csv(output_path, index=False)
        
        print(f"💾 Dataset gespeichert unter: {output_path}")
        print(f"   Größe: {len(self.df)} Zeilen, {len(self.df.columns)} Spalten")


# Test-Funktion (wird nur ausgeführt, wenn Datei direkt gestartet wird)
if __name__ == "__main__":
    print("🧪 Teste IMDBDataLoader...")
    
    # Data Loader initialisieren
    loader = IMDBDataLoader('data/IMDB Dataset.csv')
    
    # Daten laden
    df = loader.load_data()
    
    # Vorverarbeiten
    df = loader.preprocess()
    
    # Statistiken anzeigen
    loader.get_statistics()
    
    # Sample erstellen
    sample = loader.get_sample(n=200)
    
    # Verarbeitetes Dataset speichern
    loader.save_processed('data/imdb_processed.csv')
    
    print("\n✅ Test abgeschlossen!")
