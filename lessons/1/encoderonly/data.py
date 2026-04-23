# data.py
# Importa il modulo torch per creare tensori e operazioni numeriche
import torch

# Importa le classi Dataset e DataLoader da PyTorch per gestire i dati
from torch.utils.data import Dataset, DataLoader

# Importa i type hints List e Tuple per tipizzare le funzioni
from typing import List, Tuple

# Importa la classe SimpleTokenizer dal file tokenizer.py
from tokenizer import SimpleTokenizer


# ------------------------------------------------------------------
# Dataset italiano — recensioni di film
# etichetta 1 = positivo, 0 = negativo
# ------------------------------------------------------------------


# Funzione che restituisce un dataset di coppie (testo, etichetta)
def get_corpus() -> List[Tuple[str, int]]:
    # Ritorna una lista di tuple dove ogni tupla contiene una recensione e la sua etichetta
    # 1 = review positiva, 0 = review negativa
    return [
        # Ogni riga è una tupla (testo_recensione, etichetta_sentimento)
        ("il film è bellissimo davvero", 1),
        ("storia noiosa e recitazione pessima", 0),
        ("mi ha emozionato moltissimo", 1),
        ("non lo consiglio a nessuno", 0),
        ("capolavoro assoluto del cinema italiano", 1),
        ("una perdita di tempo totale", 0),
        ("attori bravissimi e regia curata", 1),
        ("trama confusa e finale deludente", 0),
        ("dialoghi brillanti e storia coinvolgente", 1),
        ("fotografia splendida e musiche bellissime", 1),
        ("personaggi piatti e sceneggiatura debole", 0),
        ("ritmo lento e storia prevedibile", 0),
        ("finale emozionante e inaspettato", 1),
        ("effetti speciali mediocri e trama assente", 0),
        ("regia magistrale e interpretazioni intense", 1),
        ("film inutile e privo di senso", 0),
    ]


# ------------------------------------------------------------------
# Dataset PyTorch
# ------------------------------------------------------------------


# Definiamo una classe che estende Dataset di PyTorch per gestire i dati
class ReviewDataset(Dataset):  # definiamo una classe che estende Dataset di PyTorch
    """
    Prende il corpus e lo trasforma in tensori
    pronti per il modello.

    Ogni campione è una tupla:
      - input_ids:      LongTensor [max_seq_len]
      - attention_mask: LongTensor [max_seq_len]
      - label:          LongTensor scalare (0 o 1)
    """

    # Metodo costruttore che inizializza il dataset
    def __init__(self, tokenizer: SimpleTokenizer, max_seq_len: int = 32):
        # Chiama il costruttore della classe padre Dataset
        super().__init__()
        # Memorizza il tokenizer per usarlo durante l'encoding dei testi
        self.tokenizer = tokenizer  # usiamo il tokenizer costruito in tokenizer.py
        # Memorizza la lunghezza massima che ogni sequenza deve avere
        self.max_seq_len = (
            max_seq_len  # lunghezza massima della sequenza (inclusi padding e [CLS])
        )
        # Carica il corpus (lista di tuple testo-etichetta) dalla funzione get_corpus()
        self.data = (
            get_corpus()
        )  # carichiamo il corpus, che è una lista di tuple (testo, etichetta)

    # questi due metodi vengono chiamati da DataLoader per iterare sul dataset (non da noi direttamente)
    # Metodo che restituisce il numero totale di campioni nel dataset
    def __len__(self) -> int:  # restituisce il numero di campioni nel dataset
        # Ritorna la lunghezza della lista self.data
        return len(self.data)

    # Metodo che restituisce un singolo campione dal dataset
    def __getitem__(self, idx: int):  # restituisce il campione alla posizione idx
        # Estrae la tupla (testo, etichetta) dalla posizione idx
        text, label = self.data[idx]
        # self.data[idx] recupera la tupla alla posizione idx
        # es. self.data[0] = ("il film è bellissimo davvero", 1)
        # text, label = ... spacchetta la tupla in due variabili separate
        # equivalente a:
        #   tupla = self.data[idx]
        #   text  = tupla[0]
        #   label = tupla[1]

        # TODO 1 ──────────────────────────────────────────────────
        # Prepara la sequenza di input per il modello.
        #
        # Passi:
        #   1. encode del testo con self.tokenizer.encode(text)
        #   2. aggiungi [CLS] all'inizio:
        #      ids = [self.tokenizer.cls_id()] + ids
        #   3. tronca se troppo lunga:
        #      ids = ids[:self.max_seq_len]
        #   4. costruisci attention_mask: lista di 1, stessa lunghezza di ids
        #   5. fai padding di ids e attention_mask fino a max_seq_len
        #      con self.tokenizer.pad_id() per ids e 0 per la maschera
        #
        # Alla fine ids e attention_mask devono avere
        # entrambi lunghezza esattamente max_seq_len.
        #
        # Input:  "il film è bello"  ->  ids grezzo: [4, 23, 7, 156]
        # Output: ids = [2, 4, 23, 7, 156, 0, 0, ...]  (lunghezza 32)
        #         mask= [1, 1,  1, 1,   1, 0, 0, ...]  (lunghezza 32)

        # !!! VIBATO - Implementazione del TODO 1

        # Codifica il testo in una lista di indici interi usando il tokenizer
        ids = self.tokenizer.encode(text)  # restituisce una lista di interi
        # Aggiunge il token speciale [CLS] all'inizio (convenzione BERT per classificazione)
        ids = [
            self.tokenizer.cls_id()
        ] + ids  # aggiungiamo l'id di [CLS] all'inizio (convenzione BERT)
        # Tronca la sequenza se è più lunga di max_seq_len
        ids = ids[: self.max_seq_len]
        # Crea una attention_mask con 1 per token reali e 0 per token di padding
        attention_mask = [1] * len(ids)

        # Calcola il numero di token di padding necessari
        padding_length = self.max_seq_len - len(ids)
        # Aggiunge gli indici di padding agli ids fino a raggiungere max_seq_len
        ids = ids + [self.tokenizer.pad_id()] * padding_length
        # Aggiunge 0 alla attention_mask per i token di padding
        attention_mask = attention_mask + [0] * padding_length

        # TODO 2 ──────────────────────────────────────────────────
        # Converti ids, attention_mask e label in tensori PyTorch.
        # Usa torch.tensor(..., dtype=torch.long) per tutti e tre.
        # Restituisci la tupla (input_ids, attention_mask, label).

        # !!! VIBATO - Implementazione del TODO 2

        # Converte la lista di indici in un tensore PyTorch di tipo long (int64)
        input_ids = torch.tensor(ids, dtype=torch.long)
        # Converte la lista della attention_mask in un tensore PyTorch di tipo long
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # Converte l'etichetta intera in un tensore PyTorch scalare di tipo long
        label = torch.tensor(label, dtype=torch.long)
        # Restituisce una tupla con i tre tensori
        return input_ids, attention_mask, label


# ------------------------------------------------------------------
# Funzione di utilità per creare tokenizer + dataloader in un colpo
# ------------------------------------------------------------------


# Funzione che costruisce tutto quello che serve per il training in una volta sola
def build_dataloader(
    max_seq_len: int = 32, batch_size: int = 4, shuffle: bool = True
) -> Tuple[SimpleTokenizer, DataLoader]:
    """
    Costruisce tokenizer, dataset e dataloader pronti per il training.
    Restituisce entrambi perché train.py ha bisogno del tokenizer
    per sapere il vocab_size da passare al modello.
    """
    # Carica il corpus di tutte le recensioni con etichette
    corpus = get_corpus()
    # Estrae solo i testi (la prima parte della tupla) dalle coppie (testo, etichetta)
    texts = [text for text, _ in corpus]

    # Crea una nuova istanza del tokenizer
    tok = SimpleTokenizer()
    # Costruisce il vocabolario dal corpus di testi
    tok.build_vocab(texts)

    # Crea un'istanza del dataset con il tokenizer e la lunghezza massima
    dataset = ReviewDataset(tok, max_seq_len=max_seq_len)
    # Crea un DataLoader che raggruppa i campioni in batch e li mescola (shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Restituisce sia il tokenizer che il dataloader
    return tok, loader


# ------------------------------------------------------------------
# Test rapido
# ------------------------------------------------------------------

# Il blocco if __name__ == "__main__" esegue il codice solo se il file
# viene eseguito direttamente, non quando viene importato come modulo
if __name__ == "__main__":
    # Chiama build_dataloader per creare il tokenizer e il dataloader
    tok, loader = build_dataloader(max_seq_len=16, batch_size=4)

    # Stampa il numero totale di token nel vocabolario
    print(f"Vocabolario: {tok.vocab_size()} token")
    # Stampa il numero di campioni nel dataset
    print(f"Dataset: {len(loader.dataset)} campioni")
    # Stampa il numero di batch nel dataloader (dataset diviso in batch)
    print(f"Batch: {len(loader)} batch da 4\n")

    # mostra il primo batch
    # next(iter(loader)) ottiene il primo batch dal dataloader
    # Spacchetta il batch in tre tensori (input_ids, attention_mask, labels)
    input_ids, attention_mask, labels = next(iter(loader))

    # Stampa la forma del tensore input_ids (numero di campioni x lunghezza sequenza)
    print(f"input_ids shape:      {input_ids.shape}")
    # Stampa la forma della attention_mask (stessa forma di input_ids)
    print(f"attention_mask shape: {attention_mask.shape}")
    # Stampa la forma dei label (numero di campioni,)
    print(f"labels shape:         {labels.shape}")
    # Stampa una riga vuota per leggibilità
    print()

    # decodifica la prima frase del batch per verifica visiva
    # input_ids[0] ottiene il primo campione del batch (forma: [max_seq_len])
    # .tolist() converte il tensore PyTorch in una lista Python
    prima_frase = input_ids[0].tolist()
    # Stampa gli indici della prima frase come lista di numeri
    print(f"Prima frase (ids):    {prima_frase}")
    # Usa il tokenizer per decodificare gli indici e ottenere il testo
    print(f"Prima frase (testo):  {tok.decode(prima_frase)}")
    # Stampa l'etichetta della prima frase (0 = negativo, 1 = positivo)
    # .item() estrae il valore numerico dal tensore scalare
    print(f"Etichetta:            {labels[0].item()}")
