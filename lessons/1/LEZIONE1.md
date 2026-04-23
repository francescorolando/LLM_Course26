# Lezione 1: Encoder-Only Transformer per Classificazione

Questa lezione introduce i modelli encoder-only attraverso un esempio pratico di classificazione binaria di recensioni di film in italiano.

## Obiettivi della Lezione

- Comprendere l'architettura dei modelli encoder-only
- Implementare un tokenizer semplice da zero
- Costruire un dataset personalizzato per il task di classificazione
- Allenare un modello Transformer per classificazione binaria
- Capire il ruolo del token [CLS] e del positional encoding

## Struttura del Progetto

```
encoderonly/
├── data.py          # Dataset e DataLoader per recensioni di film
├── model.py         # Modello encoder-only
├── tokenizer.py     # Tokenizer semplice basato su parole
├── train.py         # Script di training
├── checkpoints/     # Modelli allenati salvati
└── LEZIONE1.md        # Questa guida
```

## Componenti Principali

### 1. Tokenizer (`tokenizer.py`)

Implementa un tokenizer semplice che:

- Tokenizza il testo in parole minuscole
- Gestisce la punteggiatura separandola dalle parole
- Costruisce un vocabolario basato sulla frequenza delle parole
- Supporta token speciali: `[PAD]`, `[UNK]`, `[CLS]`

### 2. Dataset (`data.py`)

- Dataset di recensioni di film italiane (16 esempi bilanciati)
- Prepara sequenze con token [CLS] all'inizio
- Gestisce padding e attention mask
- Etichette: 1 = positivo, 0 = negativo

### 3. Modello (`model.py`)

Architettura Transformer encoder-only:

- **Embedding Layer**: Converte token IDs in vettori
- **Positional Encoding**: Aggiunge informazione di posizione usando funzioni seno/coseno
- **Transformer Encoder**: Stack di layer di attenzione multi-head
- **Classification Head**: Layer lineare che usa il token [CLS] per la predizione

### 4. Training (`train.py`)

- Loop di training con CrossEntropyLoss
- Optimizer AdamW
- Metriche: loss e accuracy per epoca
- Salvataggio del modello allenato

## Come Eseguire

### 1. Installazione Dipendenze

```
pip install torch
```

### 2. Completare i TODO

I file contengono esercizi guidati (TODO) che devono essere completati:

- `tokenizer.py`: Implementare `build_vocab()` e `encode()`
- `data.py`: Implementare `__getitem__()` per preparare i dati
- `model.py`: Implementare `__init__()` e `forward()` del modello
- `train.py`: Implementare il training loop

### 3. Eseguire il Training

```
python train.py
```

### 4. Testare il Modello

Dopo il training, il modello viene salvato in `checkpoints/encoder_classifier.pt`

## Concetti Chiave

### Encoder-Only vs Decoder-Only

- **Encoder-Only**: Ottimo per task che richiedono comprensione del contesto completo (classificazione, NER, etc.)
- **Decoder-Only**: Migliore per generazione sequenziale (GPT, autoregressive)

### Token [CLS]

- Token speciale aggiunto all'inizio di ogni sequenza
- Aggrega l'informazione dell'intera sequenza
- Usato come rappresentazione della frase per la classificazione

### Positional Encoding

- Aggiunge informazione di posizione ai token embeddings
- Usa funzioni seno/coseno a frequenze diverse
- Permette al modello di distinguere l'ordine delle parole

### Attention Mask

- Indica quali token sono reali (1) e quali sono padding (0)
- Previene che il modello presti attenzione ai token di padding

## Risultati Attesi

Dopo 20 epoche di training con i parametri attuali:

- Accuracy: ~90-100% sul dataset di training
- Loss: Convergenza sotto 0.1
- Parametri allenabili: ~50,000 circa

## Estensioni Possibili

- Aumentare il dataset con più recensioni
- Aggiungere validation set
- Implementare early stopping
- Provare diverse architetture (più layer, dimensioni maggiori)

## Note Tecniche

- **batch_first=True**: I tensori hanno shape `[batch, seq_len, d_model]`
- **norm_first=True**: Layer normalization prima dell'attenzione
- **activation="gelu"**: GELU invece di ReLU per migliore performance
- **src_key_padding_mask**: Maschera invertita per TransformerEncoder

# GUIDA BERT

https://huggingface.co/docs/transformers/model_doc/bert
