# prompting_eval.py
# Confronto strategie di prompting su dataset QA
#
# Obiettivo: misurare accuracy di zero-shot vs few-shot vs CoT
# sullo stesso set di domande

import re
import ollama
from dataset_qa import QA_DATASET

MODEL = "gemma2:2b"
N = 5  # quante domande testare: aumenta per risultati più stabili


# ------------------------------------------------------------------
# Funzioni di prompting
# ------------------------------------------------------------------


def zero_shot(question: str) -> str:
    """Nessun esempio, solo la domanda."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Answer this question with a single word or short phrase:\n{question}",
            }
        ],
    )
    return response["message"]["content"].strip()


def few_shot(question: str) -> str:
    """Tre esempi prima della domanda reale."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Answer each question with a single word or short phrase.

Q: What is the capital of Italy?
A: Rome

Q: Who wrote Hamlet?
A: Shakespeare

Q: What is the chemical symbol for gold?
A: Au

Q: {question}
A:""",
            }
        ],
    )
    return response["message"]["content"].strip()


def chain_of_thought(question: str) -> str:
    """Chiede al modello di ragionare prima di rispondere."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question step by step.
You MUST end your response with this exact format on a new line:
FINAL ANSWER: <your answer>

Question: {question}""",
            }
        ],
    )
    return response["message"]["content"].strip()


# ------------------------------------------------------------------
# Funzione di valutazione
# ------------------------------------------------------------------


def extract_answer(response: str, strategy: str) -> str:
    """
    Estrae la risposta finale dal testo del modello.
    Per CoT cerca FINAL ANSWER: ovunque nella risposta.
    Per le altre strategie usa la prima riga.
    """
    if strategy == "chainofthought":
        # cerca FINAL ANSWER: ovunque — gestisce <risposta> e testo libero
        match = re.search(r"FINAL ANSWER[:\s]+(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # fallback — prende l'ultima riga non vuota
        # re.split gestisce sia \n reali che \\n letterali
        lines = [l.strip() for l in re.split(r"\n|\\n", response) if l.strip()]
        return lines[-1] if lines else response

    return re.split(r"\n|\\n", response)[0].strip()


def is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Controlla se la risposta è corretta.
    Confronto case-insensitive e verifica se la risposta
    contiene la risposta corretta (o viceversa).
    """
    pred = prediction.lower().strip()
    truth = ground_truth.lower().strip()
    return truth in pred or pred in truth


def evaluate(strategy_fn: callable, strategy_name: str, dataset: list, n: int):
    """
    Valuta una strategia su n esempi del dataset.
    Restituisce l'accuracy.
    """
    print(f"\n{'='*50}")
    print(f"  {strategy_name}")
    print(f"{'='*50}")

    correct = 0
    for i, example in enumerate(dataset[:n]):
        question = example["question"]
        truth = example["answer"]

        response = strategy_fn(question)
        predicted = extract_answer(
            response, strategy_name.lower().replace("-", "").replace(" ", "")
        )
        ok = is_correct(predicted, truth)

        if ok:
            correct += 1

        print(f"  [{i+1}/{n}] Q: {question}")
        print(f"  Atteso:   {truth}")
        print(f"  Raw:      {response}")
        print(f"  Predetto: {predicted}")
        print(f"  {'OK' if ok else 'NO'}\n")

    accuracy = correct / n * 100
    print(f"Accuracy {strategy_name}: {correct}/{n} = {accuracy:.1f}%")
    return accuracy


# ------------------------------------------------------------------
# Confronto
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Modello: {MODEL}")
    print(f"Domande: {N}")

    acc_zero = evaluate(zero_shot, "Zero-shot", QA_DATASET, N)
    acc_few = evaluate(few_shot, "Few-shot", QA_DATASET, N)
    acc_cot = evaluate(chain_of_thought, "Chain-of-thought", QA_DATASET, N)

    # -- riepilogo ─────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RIEPILOGO")
    print("=" * 50)
    print(f"  Zero-shot:        {acc_zero:.1f}%")
    print(f"  Few-shot:         {acc_few:.1f}%")
    print(f"  Chain-of-thought: {acc_cot:.1f}%")
    print()

    best = max(
        [("Zero-shot", acc_zero), ("Few-shot", acc_few), ("CoT", acc_cot)],
        key=lambda x: x[1],
    )
    print(f"  Strategia migliore: {best[0]} ({best[1]:.1f}%)")
