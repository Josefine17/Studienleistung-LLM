# main.py

from data_loader import IMDBDataLoader
import pandas as pd

def main():
    """
    Hauptprogramm zum Vorbereiten der IMDB-Daten für Sentiment-Analyse
    """
    print("="*70)
    print(" "*15 + "IMDB SENTIMENT ANALYSIS - DATA PREPARATION")
    print("="*70)
    
    # Schritt 1: Data Loader initialisieren
    print("\n[Schritt 1/5] Initialisiere Data Loader...")
    loader = IMDBDataLoader('data/IMDB Dataset.csv')
    
    # Schritt 2: Daten laden
    print("\n[Schritt 2/5] Lade Daten...")
    try:
        df = loader.load_data()
    except FileNotFoundError as e:
        print(e)
        print("\n❌ Programm wird beendet.")
        return
    
    # Schritt 3: Daten vorverarbeiten
    print("\n[Schritt 3/5] Verarbeite Daten...")
    df = loader.preprocess()
    
    # Schritt 4: Statistiken anzeigen
    print("\n[Schritt 4/5] Zeige Statistiken...")
    loader.get_statistics()
    
    # Schritt 5: Verschiedene Samples erstellen
    print("\n[Schritt 5/5] Erstelle Samples für Experimente...")
    
    # Kleines Test-Sample (für schnelles Testen während Entwicklung)
    print("\n▶ Erstelle Test-Sample (50 Reviews)...")
    test_sample = loader.get_sample(n=50)
    test_sample.to_csv('data/test_sample_50.csv', index=False)
    print("   💾 Gespeichert: data/test_sample_50.csv")
    
    # Mittelgroßes Sample für Experimente
    print("\n▶ Erstelle Experiment-Sample (200 Reviews)...")
    experiment_sample = loader.get_sample(n=200)
    experiment_sample.to_csv('data/experiment_sample_200.csv', index=False)
    print("   💾 Gespeichert: data/experiment_sample_200.csv")
    
    # Größeres Sample für finale Evaluation
    print("\n▶ Erstelle Evaluation-Sample (1000 Reviews)...")
    eval_sample = loader.get_sample(n=1000)
    eval_sample.to_csv('data/evaluation_sample_1000.csv', index=False)
    print("   💾 Gespeichert: data/evaluation_sample_1000.csv")
    
    # Vollständiges verarbeitetes Dataset
    print("\n▶ Speichere vollständiges verarbeitetes Dataset...")
    loader.save_processed('data/imdb_processed.csv')
    
    # Zusammenfassung
    print("\n" + "="*70)
    print(" "*25 + "✅ FERTIG!")
    print("="*70)
    print("\nErstellte Dateien:")
    print("  📄 data/test_sample_50.csv           (50 Reviews - für Tests)")
    print("  📄 data/experiment_sample_200.csv    (200 Reviews - für Experimente)")
    print("  📄 data/evaluation_sample_1000.csv   (1000 Reviews - für Evaluation)")
    print("  📄 data/imdb_processed.csv           (50000 Reviews - vollständig)")
    print("\nSie können diese Dateien nun für Ihre Sentiment-Analyse verwenden!")
    print("="*70)

if __name__ == "__main__":
    main()
