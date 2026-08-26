import re
import subprocess
import sys
from llama_cpp import Llama

# 1. Załadowanie modelu z offloadingiem do Vulkana (np. 10 warstw dla GTX 660)
print("Ładowanie modelu przez Vulkan...")
llm = Llama(
    model_path="/home/animowany221/Pulpit/modele/codey3.gguf",
    n_gpu_layers=10,
    n_ctx=2048,
    verbose=False
)
print("Model gotowy!")

def execute_code(code_str):
    """Automatycznie wykonuje wygenerowany kod Pythona w podprocesie"""
    print("\n[AUTOMATYZACJA]: Wykonywanie kodu... \n" + "-"*40)
    try:
        res = subprocess.run(
            [sys.executable, "-c", code_str],
            capture_output=False,
            text=True,
            timeout=99999999999999999999999999999999999999999999999999999999999999999999999999
        )
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(f"[Błędy/Logi]:\n{res.stderr.strip()}")
        if not res.stdout and not res.stderr:
            print("[Info]: Kod wykonał się bez wyjścia tekstowego.")
    except Exception as e:
        print(f"[Błąd wykonania]: {e}")
    print("-" * 40)

def main():
    print("Zaczynamy! Wpisz zapytanie (lub 'q' aby wyjść):\n")
    while True:
        prompt = input("\nTy: ")
        if prompt.lower() in ["q", "quit"]:
            break
            
        # Format promptu dla modelu typu Instruct
        formatted = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        output = llm(formatted, max_tokens=1024, stop=["<|eot_id|>"], echo=False)
        response = output['choices'][0]['text']
        
        print(f"\nCodey:\n{response}")
        
        # Wyciąganie i automatyczne odpalanie kodu Pythona z bloku ```python ... ```
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        for code in code_blocks:
            execute_code(code)

if __name__ == "__main__":
    main()
