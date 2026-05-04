# train.py

# Entry point del progetto: avvia il training completo.
# Esegui con: python train.py

import os
import sys
import logging
from config import OUTPUT_DIR, LOGS_DIR, SEED
from data.dataset import load_data
from model.model import load_model
from training.metrics import compute_metrics
from training.trainer import build_trainer

import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Fissa il seed per la riproducibilità.
    Va impostato su tutte le librerie che usano numeri casuali.
    Senza questo ogni run produce risultati leggermente diversi.
    """
    random.seed(seed)
    # seed per la libreria standard Python — usata in alcune operazioni
    # di shuffling e campionamento

    np.random.seed(seed)
    # seed per numpy — usato nelle metriche e nel preprocessing

    torch.manual_seed(seed)
    # seed per PyTorch — usato nell'inizializzazione dei pesi
    # e nel dropout

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # seed per tutte le GPU — necessario se usi più GPU


def main():

    # -- 0. riproducibilità ---------------------------------------
    set_seed(SEED)
    print(f"Seed: {SEED}")

    # -- 0.5. crea cartella e file di log ---------------------------
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "training.log")

    # configura il logger
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Inizio training - Seed: {SEED}")
    print(f"Log file: {log_file}")

    # -- 1. dati --------------------------------------------------
    print("\n--- Caricamento dataset ---")
    tokenized_dataset, tokenizer = load_data()
    print(f"Train:      {len(tokenized_dataset['train'])} esempi")
    print(f"Validation: {len(tokenized_dataset['validation'])} esempi")
    logger.info(
        f"Dataset caricato - Train: {len(tokenized_dataset['train'])}, Val: {len(tokenized_dataset['validation'])}"
    )

    # -- 2. modello -----------------------------------------------
    print("\n--- Caricamento modello ---")
    model = load_model()

    # -- 3. trainer -----------------------------------------------
    print("\n--- Configurazione trainer ---")
    trainer = build_trainer(
        model=model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -- 4. training ----------------------------------------------
    print("\n--- Avvio training ---")
    logger.info("Inizio training")
    trainer.train()
    logger.info("Training completato")
    # il Trainer stampa automaticamente:
    # - la loss ad ogni LOGGING_STEPS step
    # - le metriche di valutazione alla fine di ogni epoca
    # - il tempo stimato rimanente

    # -- 5. valutazione finale ------------------------------------
    print("\n--- Valutazione finale ---")
    risultati = trainer.evaluate()
    print(f"Accuracy finale: {risultati['eval_accuracy']:.4f}")
    logger.info(f"Accuracy finale: {risultati['eval_accuracy']:.4f}")

    # -- 6. salva il modello finale -------------------------------
    print(f"\n--- Salvataggio modello ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Modello salvato in: {OUTPUT_DIR}")
    print(f"Modello salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
