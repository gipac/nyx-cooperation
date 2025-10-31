# 🚀 Quick Start: Ricostruzione NYX con LLM

## Setup in 3 Passi

### 1️⃣ Verifica Ollama

```bash
# Su Windows PowerShell o Linux terminal
ollama list
```

Se non vedi output, installa Ollama: https://ollama.com/download

### 2️⃣ Scarica Mistral (3-5 min)

```bash
ollama pull mistral-7b-instruct
```

### 3️⃣ Esegui Test Veloce (5 min)

```bash
cd nyx-cooperation
python experiments_llm/test_models_bias.py --models mistral-7b-instruct --trials 10
```

---

## 🎯 Esperimento Completo

### Opzione A: Veloce (2-3 ore)

```bash
python experiments_llm/run_historical_reconstruction.py \
    --model mistral-7b-instruct \
    --fast
```

### Opzione B: Completo (6-8 ore)

```bash
python experiments_llm/run_historical_reconstruction.py \
    --model mistral-7b-instruct
```

---

## 📊 Risultati

I risultati saranno salvati in:
- `results_llm/reconstruction_mistral-7b-instruct.json`
- `results_llm/reconstruction.log`

### Analizza Risultati

```python
import json

with open('results_llm/reconstruction_mistral-7b-instruct.json') as f:
    results = json.load(f)

# Summary delle scoperte validate
print(json.dumps(results['summary']['key_discoveries'], indent=2))
```

---

## ⚠️ Troubleshooting

### "Ollama not available"

```bash
# Windows: Apri Ollama app dal menu Start
# Linux:
sudo systemctl start ollama
```

### "Model not found"

```bash
ollama pull mistral-7b-instruct
```

### Troppo lento?

Usa un modello più piccolo:
```bash
ollama pull phi-3.5-mini
python experiments_llm/run_historical_reconstruction.py --model phi-3.5-mini --fast
```

---

## 🎓 Cosa Aspettarsi

### Le 7 Fasi

1. **Fase 0**: Baseline (~5 min)
2. **Fase 1**: Single Bit Theory (~15 min)
3. **Fase 2**: Multi-Bit Consciousness (~20 min)
4. **Fase 3**: Memory Effect (~15 min)
5. **Fase 4**: Network Effect (~25 min)
6. **Fase 5**: Formula Synthesis (~30 min)
7. **Fase 6**: 80/20 Validation (~5 min)

**Totale Fast Mode**: ~2 ore
**Totale Full Mode**: ~6-8 ore

### Output Durante Esecuzione

```
INFO - PHASE 1: SINGLE BIT THEORY
INFO - Testing 0-bit consciousness...
INFO - LLM Agent llm_agent_0: 0-bit, model=mistral-7b-instruct
INFO - Running 25 LLM interaction cycles with 4 agents
INFO - Cycle 0/25
...
INFO - ✅ Phase 1 Complete: 45.2% cooperation jump
INFO -    Single Bit Theory: ✓ VALIDATED
```

---

## 📈 Interpretazione Risultati

### ✅ Successo

Se vedi questi simboli alla fine:
```
✓ single_bit_theory: True
✓ optimal_2_bit: True
✓ minimum_viable_society: True
✓ formula_accuracy > 0.85
✓ 80_20_law: True
```

**→ Le leggi NYX si applicano anche agli LLM reali!**

### ⚠️ Parziale

Se alcuni test falliscono:
- Normale con fast mode (pochi dati)
- Prova full mode
- Oppure modello diverso

### ❌ Fallimento

Se molti test falliscono:
- Potrebbe essere bias RLHF troppo forte
- Prova con temperature più alta
- Oppure modello base (non instruct)

---

## 🔬 Domande di Ricerca

Dopo gli esperimenti, chiediti:

1. **La Single Bit Theory funziona con LLM reali?**
   - Check: `phase_1.jump_validated`

2. **2-bit è davvero ottimale?**
   - Check: `phase_2.optimal_is_2_bit`

3. **La formula predice accuratamente?**
   - Check: `phase_5.overall_accuracy`

4. **La coscienza pesa 80%?**
   - Check: `phase_6.consciousness_percentage`

---

## 🎯 Next Steps

Dopo la ricostruzione:

1. **Confronta con simulazione originale**
   ```bash
   python scripts/reproduce_paper_results.py --fast
   ```

2. **Testa altri modelli**
   ```bash
   ollama pull llama-3.2-3b-instruct
   python experiments_llm/run_historical_reconstruction.py \
       --model llama-3.2-3b-instruct --fast
   ```

3. **Analizza differenze**
   - Quale modello segue meglio le leggi NYX?
   - Quale ha più bias?
   - LLM vs simulazione: differenze?

---

**Pronto per iniziare? Esegui:**

```bash
python experiments_llm/test_models_bias.py --models mistral-7b-instruct --trials 10
```

**Buona scoperta! 🔬✨**
