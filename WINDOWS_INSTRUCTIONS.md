# 🪟 NYX Experiments - Windows Instructions

Poiché il container Linux non può accedere a Ollama su Windows, usa questo script standalone.

## 🚀 Setup Rapido

### 1. Apri PowerShell nella cartella del progetto

```powershell
cd C:\path\to\nyx-cooperation
```

### 2. Verifica Ollama

```powershell
ollama list
```

Dovresti vedere `mistral-7b-instruct` nella lista.

### 3. Test Connessione (30 secondi)

```powershell
python run_nyx_windows.py --test-connection
```

Output atteso:
```
✅ Ollama connected!
✅ All systems operational!
```

---

## 🧪 Esperimenti

### Test Veloce Bias (2-3 minuti)

```powershell
python run_nyx_windows.py --test-bias --trials 10
```

Questo testa il bias cooperativo del modello a diversi livelli di coscienza.

### Fase Singola (15-30 minuti)

```powershell
# Esempio: Solo Single Bit Theory
python run_nyx_windows.py --phase 1 --fast
```

Fasi disponibili:
- **0**: Baseline (no coscienza)
- **1**: Single Bit Theory
- **2**: Multi-Bit Consciousness
- **3**: Memory Effect
- **4**: Network Effect
- **5**: Formula Synthesis
- **6**: 80/20 Law Validation

### Ricostruzione Completa (2-3 ore fast, 6-8 ore full)

**Fast Mode (raccomandato per primo test):**
```powershell
python run_nyx_windows.py --fast
```

**Full Mode (più dati, più accurato):**
```powershell
python run_nyx_windows.py
```

---

## 📊 Risultati

I risultati saranno salvati in:
- `results_llm/reconstruction_mistral-7b-instruct.json` (completo)
- `results_llm/phase_N_result.json` (singole fasi)
- `nyx_windows_run.log` (log dettagliato)

### Analizza Risultati

```powershell
python
```

```python
import json

# Carica risultati
with open('results_llm/reconstruction_mistral-7b-instruct.json') as f:
    results = json.load(f)

# Scoperte validate
print("Scoperte NYX Validate:")
for discovery, validated in results['summary']['key_discoveries'].items():
    symbol = "✓" if validated else "✗"
    print(f"  {symbol} {discovery}")

# Dettagli Single Bit Theory
phase1 = results['phase_1']
print(f"\nSingle Bit Jump: {phase1['cooperation_jump']:.1%}")
print(f"Validated: {phase1['jump_validated']}")
```

---

## ⚙️ Opzioni Avanzate

### Usa Modello Diverso

```powershell
# Llama 3.2 (più veloce)
ollama pull llama-3.2-3b-instruct
python run_nyx_windows.py --model llama-3.2-3b-instruct --fast

# Phi 3.5 (molto veloce)
ollama pull phi-3.5-mini
python run_nyx_windows.py --model phi-3.5-mini --fast
```

### Più Trial per Bias Test

```powershell
python run_nyx_windows.py --test-bias --trials 50
```

---

## 🐛 Troubleshooting

### "Ollama not available"

1. Apri Ollama app dal menu Start
2. Verifica con: `ollama list`
3. Se necessario: `ollama serve`

### "Model not found"

```powershell
ollama pull mistral-7b-instruct
```

### Script troppo lento

Usa un modello più piccolo:
```powershell
ollama pull phi-3.5-mini
python run_nyx_windows.py --model phi-3.5-mini --fast
```

### Out of Memory

Chiudi altre applicazioni e riprova con `--fast`

---

## 📈 Cosa Aspettarsi

### Output Durante Esecuzione

```
NYX HISTORICAL RECONSTRUCTION - FULL RUN
============================================================
Model: mistral-7b-instruct
Mode: FAST
Start: 2025-10-31 18:00:00
============================================================

============================================================
PHASE 0: BASELINE (No Consciousness)
============================================================
Creating 4 LLM agents with 0-bit consciousness
Running 20 LLM interaction cycles with 4 agents
Cycle 0/20
...
✅ Phase 0 Complete: 5.2% cooperation

============================================================
PHASE 1: SINGLE BIT THEORY
============================================================
Testing 0-bit consciousness...
Testing 1-bit consciousness...
✅ Phase 1 Complete: 45.3% cooperation jump
   Single Bit Theory: ✓ VALIDATED
```

### Tempi Stimati

| Configurazione | Fast Mode | Full Mode |
|----------------|-----------|-----------|
| Mistral 7B (CPU) | 2-3h | 6-8h |
| Mistral 7B (GPU) | 1-1.5h | 3-4h |
| Llama 3.2 3B | 1.5-2h | 4-5h |
| Phi 3.5 Mini | 1-2h | 3-4h |

---

## 🎯 Domande di Ricerca

Dopo gli esperimenti, analizza:

1. **Single Bit Theory funziona con LLM reali?**
   - Check: `phase_1.jump_validated`
   - Atteso: jump > 40%

2. **2-bit è ottimale?**
   - Check: `phase_2.optimal_is_2_bit`
   - Atteso: True

3. **Formula NYX predice accuratamente?**
   - Check: `phase_5.overall_accuracy`
   - Atteso: > 85%

4. **Coscienza pesa 80%?**
   - Check: `phase_6.consciousness_percentage`
   - Atteso: ~80%

---

## 📝 Note

- Lo script usa `localhost:11434` per Ollama
- I risultati possono variare tra run (LLM stocastici)
- Temperature = 0.7 per bilanciare determinismo e variabilità
- Tutti i modelli testati sono instruction-tuned (hanno bias RLHF)

---

## 🤝 Prossimi Passi

1. **Test veloce**: `python run_nyx_windows.py --test-bias`
2. **Una fase**: `python run_nyx_windows.py --phase 1 --fast`
3. **Completo**: `python run_nyx_windows.py --fast`
4. **Analizza** e confronta con simulazione originale

---

**Pronto? Inizia con:**

```powershell
python run_nyx_windows.py --test-connection
```

**Buona scoperta! 🔬✨**
