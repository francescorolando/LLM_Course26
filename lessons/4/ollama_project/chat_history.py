# chat_history.py
# Conversazione multi-turn con memoria
#
# Obiettivo: capire come funziona la memoria nei LLM
# e come costruire una conversazione con contesto

import ollama
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL

# ------------------------------------------------------------------
# Come funziona la memoria negli LLMs
# ------------------------------------------------------------------
# Gli LLMs non hanno memoria persistente tra chiamate diverse.
# Ogni chiamata è indipendente e il modello non ricorda nulla.
#
# Per simulare una conversazione, devi passare l'intera storia
# dei messaggi ad ogni chiamata:
#
# chiamata 1: [user: "Ciao"]
# chiamata 2: [user: "Ciao", assistant: "Ciao!", user: "Come ti chiami?"]
# chiamata 3: [user: "Ciao", assistant: "Ciao!", user: "Come ti chiami?",
#              assistant: "Mi chiamo Gemma", user: "Quanti anni hai?"]
#
# Il modello "ricorda" solo perché gli passi tutto ogni volta.

# ------------------------------------------------------------------
# 1. Dimostrazione — senza memoria
# ------------------------------------------------------------------

print("=" * 50)
print("  1. SENZA MEMORIA — ogni chiamata è indipendente")
print("=" * 50)

r1 = ollama.chat(
    model=MODEL, messages=[{"role": "user", "content": "Il mio nome è Giulia."}]
)
print(f"Turno 1: {r1['message']['content']}")

r2 = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Come mi chiamo?"}],
    # nota: non passiamo il messaggio precedente
    # il modello non sa chi sei
)
print(f"Turno 2: {r2['message']['content']}")
# il modello risponderà che non sa come ti chiami


# ------------------------------------------------------------------
# 2. Dimostrazione — con memoria
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  2. CON MEMORIA — passiamo la storia completa")
print("=" * 50)

history = []


def chat(user_message: str) -> str:
    """
    Invia un messaggio mantenendo la storia della conversazione.
    Ogni chiamata aggiunge il messaggio utente e la risposta
    del modello alla history.
    """
    history.append({"role": "user", "content": user_message})

    response = ollama.chat(model=MODEL, messages=history)
    risposta = response["message"]["content"]

    history.append({"role": "assistant", "content": risposta})

    return risposta


r1 = chat("Il mio nome è Giulia.")
print(f"Turno 1 — utente: 'Il mio nome è Giulia.'")
print(f"Turno 1 — modello: {r1}")

r2 = chat("Come mi chiamo?")
print(f"\nTurno 2 — utente: 'Come mi chiamo?'")
print(f"Turno 2 — modello: {r2}")

r3 = chat("E qual era il contesto della nostra conversazione?")
print(f"\nTurno 3 — utente: 'E qual era il contesto della nostra conversazione?'")
print(f"Turno 3 — modello: {r3}")


# ------------------------------------------------------------------
# 3. Chatbot interattivo con system prompt
# ------------------------------------------------------------------

print("\n" + "=" * 50)
print("  3. CHATBOT INTERATTIVO")
print("     digita 'exit' per uscire, 'reset' per ricominciare")
print("=" * 50 + "\n")

# TODO ─────────────────────────────────────────────────────────────
# Completa il chatbot interattivo:
#
# 1. Inizializza la history con un system prompt a tua scelta
#    es. un assistente che risponde solo in italiano,
#    o un esperto di cinema, o un tutor di matematica
#
# 2. Implementa il loop while True:
#    - leggi l'input dell'utente con input()
#    - gestisci "exit" per uscire
#    - gestisci "reset" per svuotare la history
#      (ma mantieni il system prompt!)
#    - chiama ollama.chat() con la history aggiornata
#    - stampa la risposta con streaming (stream=True)
#
# Struttura suggerita:
#
# system_prompt = "..."
# history = [{"role": "system", "content": system_prompt}]
#
# while True:
#     user_input = input("Tu: ").strip()
#     if user_input.lower() == "exit":
#         break
#     if user_input.lower() == "reset":
#         history = [{"role": "system", "content": system_prompt}]
#         print("Conversazione resettata.")
#         continue
#
#     history.append({"role": "user", "content": user_input})
#
#     print("Assistente: ", end="", flush=True)
#     ...
# ─────────────────────────────────────────────────────────────────────

# 1. Inizializza la history con un system prompt
system_prompt = (
    "Sei un assistente esperto di machine learning e LLM. "
    "Rispondi sempre in italiano, in modo chiaro e conciso. "
    "Se l'utente ti chiede qualcosa fuori dal tuo ambito, "
    "suggerisci gentilmente di riportare la conversazione "
    "su argomenti di machine learning."
)
history = [{"role": "system", "content": system_prompt}]

# 2. Implementa il loop while True con gestione comandi
while True:
    # - leggi l'input dell'utente con input()
    user_input = input("Tu: ").strip()

    # - gestisci "exit" per uscire
    if user_input.lower() == "exit":
        print("Arrivederci!")
        break

    # - gestisci "reset" per svuotare la history (ma mantieni il system prompt!)
    if user_input.lower() == "reset":
        history = [{"role": "system", "content": system_prompt}]
        print("Conversazione resettata.\n")
        continue

    # Ignora input vuoto
    if not user_input:
        continue

    # - chiama ollama.chat() con la history aggiornata
    history.append({"role": "user", "content": user_input})

    # - stampa la risposta con streaming (stream=True)
    print("Assistente: ", end="", flush=True)
    response_text = ""
    for chunk in ollama.chat(model=MODEL, messages=history, stream=True):
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        response_text += content
    print()  # newline dopo la risposta

    # Aggiungi la risposta completa alla history
    history.append({"role": "assistant", "content": response_text})
