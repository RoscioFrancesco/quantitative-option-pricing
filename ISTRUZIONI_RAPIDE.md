# Avvio rapido

## Avvio consigliato

Apri il terminale nella cartella del progetto e lancia:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Puoi anche lanciare:

```bash
streamlit run quantitative_option_pricing.py
```

`quantitative_option_pricing.py` richiama semplicemente `app.py`, quindi entrambi i comandi funzionano.

Su macOS puoi usare `bash scripts/setup_mac.sh` una sola volta e poi aprire `run.command`. Su Windows esegui prima `scripts/setup_windows.bat`, poi `scripts/run_windows.bat`.

## Se Streamlit è già aperto

Ferma il server con `CTRL + C`, poi rilancia:

```bash
streamlit run app.py
```
