Update:

Sentiment Analyse mit Kaggle Datenset https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Erkennung der Stimmung von IMDB-Rezensionen
- Beispiel-Prompt: Bewerte die Stimmung dieser Rezension: positiv, neutral oder negativ
- mit verschiedenen Prompt-Strategien (Zero-Shot vs. Few-Shot mit Beispielen)
- Model: Llama 3.2





# Studienleistung-LLM

Modell: DeepSeek-V3.1

Aufgabenstellung
- Es soll geschaut werden wie verschiedene Punkte aus dem Datensatz zueinander stehen 
-   z.B. Genre Eingrenzung, zeitliche Eingrenzung, Umsatz, Release Date, Bewertung
- Verwendung von TMDB Datensatz aus kaggle (https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies, https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) ODER per API-Schnittstelle
- Das Projekt soll untersuchen, wie gut das LLM Zusammenhänge erkennt und Fragen beantworten kann, dazu sollen Prognosen abgeleitet und mit den Ergebnissen verglichen werden

Verfahrensweise 
- Ziele definieren & Prognosen aufstellen
- Überlegung welche Daten aus TMDB verwendet werden sollen
- API-Key einrichten & Python-Skript zum Abruf der Daten schreiben
- Open-LLaMa lokal einrichten & Schnittstelle schreiben
- Promptvorlagen definieren
- LLM mit verschiedenen Fragestellungen testen
- Ergebnisse dokumentieren und mit Prognosen vergleichen
- Vergleich mit klassischen Machine-Learning-Methoden
- Zunächst nur mit deutschen Prompts, aber zum Vergleich auch englische nutzen
- Resultate ziehen & Projekt bewerten

Ideen, was analysiert werden könnte:
--> Daten sollen miteinander Kombiniert werden und daraus Rückschlüsse gezogen werden
- Wie häufig ein Wort des Titels in der Filmbeschreibung vorkommt (Genre Eingrenzung)
- Wie beliebt ein Filmgenre ist, gemessen daran, wie viele Filme es in dem Genre gibt (zeitliche Eingrenzung)
- Spiegelt sich deine gute Bewertung des Films in dem Umsatz, der generiert wurde wider?
- Gibt es eine besondere Häufung von Genres zu einem bestimmten Release-Datum?
- Gibt es eine Häufung von Umsatz, der zu bestimmten Release-Zeiträumen generiert wird? Bsp. Mehr Geld wird eingenommen bei Filmen im Herbst, als im Sommer. 
- Werden Filme, die populärer sind, häufiger gevotet? Dasselbe gilt für Umsatz (mehr Umsatz = mehr gevoted)
- Wie ist die Bewertung in unterschiedlichen Genres?
- Trendanalyse


Arbeitsteilung 
NJ: TMDB - API - Anbindung, Auswertung/Interpretation
JH: Prompts schreiben, Prompt Analyse
