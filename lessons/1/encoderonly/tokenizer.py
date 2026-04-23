# tokenizer.py
# Importa il modulo 're' per gestire le espressioni regolari (pattern matching sul testo)
import re  # espressioni regolari

# Importa Counter da collections per contare le frequenze dei token
from collections import Counter

# Importa il type hint List per indicare liste di elementi tipizzati
from typing import List


class SimpleTokenizer:
    """
    Tokenizer minimale a parole intere.
    Non è sofisticato: non gestisce le parole composte, i prefissi/suffissi,
    o le parole fuori vocabolario in modo intelligente, ma è semplice e trasparente
    """

    # Definisce il token speciale PAD (padding) - usato per riempire sequenze corte
    PAD = "[PAD]"
    # Definisce il token speciale UNK (unknown) - usato per parole sconosciute
    UNK = "[UNK]"
    # Definisce il token speciale CLS (classification) - va all'inizio della sequenza
    CLS = "[CLS]"

    # Metodo costruttore che inizializza gli oggetti della classe
    def __init__(self):
        # Crea un dizionario vuoto dove le parole saranno chiavi e gli indici i valori
        self.vocab = {}  # parola  -> indice
        # Crea un dizionario inverso dove gli indici saranno chiavi e le parole i valori
        self.inv_vocab = {}  # indice  -> parola

    # ------------------------------------------------------------------
    # 1. Pre-processing del testo grezzo
    # ------------------------------------------------------------------

    # Metodo che trasforma una stringa in una lista di token (parole)
    def _tokenize(self, text: str) -> List[str]:
        """
        Trasforma una stringa in una lista di token (parole).

        Passi:
          1. lowercase
          2. separa la punteggiatura dal testo con uno spazio
             (così "bello!" diventa ["bello", "!"] e non ["bello!"])
          3. split sugli spazi
          4. rimuove token vuoti

        Nota: gli apostrofi italiani vengono gestiti separando
        la parola in due token: "l'uomo" -> ["l'", "uomo"].
        """
        # Converte tutto il testo in minuscolo per normalizzazione
        text = text.lower()
        # Usa la regex per inserire spazi intorno alla punteggiatura (,.!?;:"-)
        # Il gruppo (.) cattura il carattere di punteggiatura e r" \1 " lo circonda di spazi
        text = re.sub(r"([,.!?;:()\"\-])", r" \1 ", text)
        # Divide il testo sugli spazi bianchi e mantiene solo i token non vuoti
        tokens = [t for t in text.split() if t]
        # Ritorna la lista di token (parole)
        return tokens

    # ------------------------------------------------------------------
    # 2. Costruzione del vocabolario
    # ------------------------------------------------------------------

    # Metodo per costruire il vocabolario a partire da una lista di frasi
    def build_vocab(self, texts: List[str], max_vocab: int = 10000) -> None:
        """
        Costruisce il vocabolario a partire da una lista di frasi.

        Ordine degli indici:
          0 -> [PAD]
          1 -> [UNK]
          2 -> [CLS]
          3, 4, 5, ... -> parole ordinate per frequenza (dalla più comune)

        Perché i token speciali vengono prima?
        Convenzione universale — indici bassi e fissi li rendono
        facili da riconoscere e da escludere dal calcolo della loss.
        """
        # TODO 1 ──────────────────────────────────────────────────────
        # Conta la frequenza di ogni token nel corpus.
        # Suggerimento: usa Counter e il metodo _tokenize su ogni frase.
        # Poi tieni solo i max_vocab - 3 token più frequenti
        # (sottrai 3 per fare spazio ai token speciali).
        #
        # Alla fine costruisci self.vocab e self.inv_vocab.
        # Ricorda: i token speciali vanno agli indici 0, 1, 2.
        #
        # Input:  texts = ["il film è bello", "non mi è piaciuto", ...]
        # Output: self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
        #                       "il": 3, "film": 4, ...}
        """ token_freq = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            token_freq.update(tokens)

        most_common = token_freq - most_common(max_vocab - 3)

        self.vocab = {self.PAD: 0, self.UNK: 1, self.CLS: 2}
        for i, (token, _) in enumerate(most_common, start=3):
            self.vocab[token] = i

        self.inv_vocab = {v: k for k, v in self.vocab.items()} """

        # !!! VIBATO - Codice funzionante che implementa il TODO 1

        # Crea un Counter (dizionario) che conterà le frequenze di ogni token
        token_freq = Counter()
        # Itera su ogni frase nella lista di testi
        for text in texts:
            # Tokenizza la frase attuale usando il metodo _tokenize
            tokens = self._tokenize(text)
            # Aggiorna il Counter con i nuovi token (incrementa i conteggi)
            token_freq.update(tokens)

        # Prende i (max_vocab - 3) token più comuni da token_freq
        # Sottrae 3 perché riserva spazio ai token speciali [PAD], [UNK], [CLS]
        most_common_tokens = token_freq.most_common(max_vocab - 3)

        # Inizializza il vocabolario con i token speciali agli indici 0, 1, 2
        self.vocab = {self.PAD: 0, self.UNK: 1, self.CLS: 2}
        # Itera sui token più comuni e assegna loro indici a partire da 3
        for i, (token, _) in enumerate(most_common_tokens, start=3):
            # Aggiunge il token al vocabolario con indice i (i aumenta ad ogni iterazione)
            self.vocab[token] = i

        # Crea il vocabolario inverso scambiando chiavi e valori
        # Ora inv_vocab mappa indici -> parole (per il decode)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ------------------------------------------------------------------
    # 3. Encode e decode
    # ------------------------------------------------------------------

    # Metodo che converte il testo in una lista di indici interi
    def encode(self, text: str) -> List[int]:
        """
        Testo -> lista di interi.
        Le parole fuori vocabolario diventano l'indice di [UNK].

        Nota: NON aggiunge [CLS] qui — lo fa data.py,
        così la responsabilità è chiara: il tokenizer traduce,
        data.py prepara le sequenze per il modello.
        """
        # TODO 2 ──────────────────────────────────────────────────────
        # Tokenizza il testo con _tokenize, poi mappa ogni token
        # al suo indice in self.vocab.
        # Se il token non è nel vocabolario, usa l'indice di [UNK].
        #
        # Input:  "il film è bello"
        # Output: [3, 4, 1, 5]   (esempio — gli indici dipendono dal vocab)

        # !!! VIBATO - Codice funzionante che implementa il TODO 2

        # Chiama _tokenize per ottenere la lista di token dalla stringa
        tokens = self._tokenize(text)

        # Mappa ogni token al suo indice nel vocabolario
        # Se il token NON è in vocab, usa .get() che ritorna l'indice di [UNK] come default
        return [self.vocab.get(t, self.vocab[self.UNK]) for t in tokens]

    # Metodo che converte una lista di indici interi nel testo originale
    def decode(self, ids: List[int]) -> str:
        """
        Lista di interi -> testo.
        Usato principalmente per debug: permette di leggere
        cosa sta "vedendo" il modello dopo tokenizzazione e padding.
        """
        # Mappa ogni indice al token corrispondente usando il vocabolario inverso
        # Se l'indice non è trovato, usa [UNK] come fallback
        tokens = [self.inv_vocab.get(i, self.UNK) for i in ids]
        # Unisce i token con spazi per formare una stringa
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # 4. Proprietà utili
    # ------------------------------------------------------------------

    # Metodo che ritorna la dimensione del vocabolario (numero totale di token)
    def vocab_size(self) -> int:
        # Ritorna il numero di elementi nel dizionario vocab
        return len(self.vocab)

    # Metodo che ritorna l'indice del token PAD
    def pad_id(self) -> int:
        # Ritorna l'indice associato al token speciale [PAD] (che è 0)
        return self.vocab[self.PAD]

    # Metodo che ritorna l'indice del token CLS
    def cls_id(self) -> int:
        # Ritorna l'indice associato al token speciale [CLS] (che è 2)
        return self.vocab[self.CLS]


# ------------------------------------------------------------------
# Test rapido — esegui questo file direttamente per verificare
# ------------------------------------------------------------------
# Il blocco if __name__ == "__main__" esegue il codice solo se il file
# viene eseguito direttamente, non quando viene importato come modulo
if __name__ == "__main__":
    # Definisce una lista di frasi di esempio per il test del tokenizer
    corpus = [
        "il film è bellissimo davvero",
        "storia noiosa e recitazione pessima",
        "mi ha emozionato molto",
        "non lo consiglio a nessuno",
        "capolavoro assoluto del cinema italiano",
        "una perdita di tempo totale",
        "attori bravissimi e regia curata",
        "trama confusa e finale deludente",
    ]

    # Crea un'istanza del tokenizer
    tok = SimpleTokenizer()
    # Costruisce il vocabolario usando le frasi del corpus
    tok.build_vocab(corpus)

    # Stampa la dimensione totale del vocabolario (numero di token unici)
    print(f"Vocabolario: {tok.vocab_size()} token")
    # Stampa i primi 10 elementi del vocabolario (i token speciali + i più frequenti)
    print(f"Prime 10 voci: {list(tok.vocab.items())[:10]}")
    # Stampa una riga vuota per leggibilità
    print()

    # Definisce una frase di test
    frase = "il film è bellissimo"
    # Codifica la frase convertendola in indici interi
    ids = tok.encode(frase)
    # Stampa il risultato dell'encoding (la frase come lista di indici)
    print(f"encode('{frase}') -> {ids}")
    # Decodifica gli indici per verificare che si possa tornare al testo (per debug)
    print(f"decode({ids})     -> '{tok.decode(ids)}'")
    # Stampa una riga vuota per leggibilità
    print()

    # Definisce una frase con una parola sconosciuta (non nel corpus)
    ids_unk = tok.encode("questo film è fantasmagorico")
    # Stampa il risultato dell'encoding con parole sconosciute
    print(f"encode con parola sconosciuta -> {ids_unk}")
    # Decodifica per mostrare come viene rappresentata la parola sconosciuta
    print(f"decode -> '{tok.decode(ids_unk)}'")
