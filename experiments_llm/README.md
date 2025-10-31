# NYX Historical Reconstruction with Real LLMs

Ricostruzione completa del percorso scientifico NYX usando **veri modelli di linguaggio** invece di agenti simulati.

## 🎯 Obiettivo

Testare se le **leggi matematiche NYX** si applicano a intelligenze artificiali reali (LLM), non solo a simulazioni deterministiche.

## 📋 Le 7 Fasi della Scoperta

### **Fase 0: Baseline**
- **Test**: Cooperazione senza meta-cognizione
- **Atteso**: ~0-10% cooperazione casuale

### **Fase 1: Single Bit Theory** 🔥
- **Test**: ROI awareness causa salto quantico?
- **Atteso**: 0% → 67.5% cooperazione

### **Fase 2: Multi-Bit Consciousness**
- **Test**: 1-bit vs 2-bit vs 3-bit
- **Atteso**: Ottimale a 2-bit, rendimenti decrescenti a 3-bit

### **Fase 3: Memory Effect**
- **Test**: Sweet spot memoria
- **Atteso**: Picco a ~10 pattern

### **Fase 4: Network Effect**
- **Test**: Minimum Viable Society
- **Atteso**: Soglia a 4 agenti

### **Fase 5: Formula Synthesis**
- **Test**: C = 0.1×N + 0.1×M + 0.8×A
- **Atteso**: >85% accuratezza

### **Fase 6: 80/20 Law**
- **Test**: Coscienza (A) domina con 80%
- **Atteso**: A=80%, N+M=20%

---

## 🚀 Setup

### 1. Installa Ollama (se non l'hai già)

**Windows:**
```powershell
# Scarica da https://ollama.com/download
# Oppure con winget:
winget install Ollama.Ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Scarica i Modelli

```bash
# Modello principale (raccomandato)
ollama pull mistral-7b-instruct

# Modelli alternativi
ollama pull llama-3.2-3b-instruct
ollama pull phi-3.5-mini

# Modello grande (validazione finale)
ollama pull gpt-oss-20b  # Solo se hai molto tempo e GPU potente
```

### 3. Installa Dipendenze Python

```bash
cd /path/to/nyx-cooperation
pip install numpy pandas requests
```

---

## 🧪 Esecuzione Esperimenti

### Test 1: Verifica Bias Modelli (30 min)

Misura il bias cooperativo intrinseco dei modelli:

```bash
python experiments_llm/test_models_bias.py \
    --models mistral-7b-instruct llama-3.2-3b-instruct phi-3.5-mini \
    --trials 20
```

**Output**: `results_llm/model_bias_results.json`

### Test 2: Baseline Simulata (2 min)

Esegui simulazione originale per confronto:

```bash
python scripts/reproduce_paper_results.py --fast --test 1
```

### Test 3: Ricostruzione Storica Completa (3-8 ore)

Esegui tutte le 7 fasi con LLM reali:

```bash
# Modalità veloce (2-3 ore)
python experiments_llm/run_historical_reconstruction.py \
    --model mistral-7b-instruct \
    --fast

# Modalità completa (6-8 ore)
python experiments_llm/run_historical_reconstruction.py \
    --model mistral-7b-instruct
```

**Output**: `results_llm/reconstruction_mistral-7b-instruct.json`

### Test 4: Fase Singola

Esegui solo una fase specifica:

```bash
# Esempio: Solo Single Bit Theory
python experiments_llm/run_historical_reconstruction.py \
    --model mistral-7b-instruct \
    --phase 1 \
    --fast
```

---

## 📊 Analisi Risultati

### Confronto Simulazione vs LLM

```python
import json

# Carica risultati simulazione originale
with open('reproduction_results/complete_reproduction_results.json') as f:
    sim_results = json.load(f)

# Carica risultati LLM
with open('results_llm/reconstruction_mistral-7b-instruct.json') as f:
    llm_results = json.load(f)

# Confronta accuratezze
print("Formula Accuracy:")
print(f"  Simulazione: {sim_results['paper_reproduction_summary']['overall_accuracy']:.1%}")
print(f"  LLM:         {llm_results['phase_5']['overall_accuracy']:.1%}")
```

### Validazione Scoperte Chiave

```python
# Verifica se le leggi NYX si applicano agli LLM
summary = llm_results['summary']['key_discoveries']

print("Scoperte Validate:")
for discovery, validated in summary.items():
    symbol = "✓" if validated else "✗"
    print(f"  {symbol} {discovery}")
```

---

## ⚙️ Configurazione Ollama

### Se Ollama è su Windows e Claude Code su WSL

1. **Windows**: Avvia Ollama (automatico all'avvio)

2. **WSL**: Usa l'IP dell'host Windows:

```bash
# Trova IP Windows
ipconfig | findstr IPv4  # Da PowerShell Windows

# Esegui con IP custom
python experiments_llm/run_historical_reconstruction.py \
    --ollama-url http://192.168.x.x:11434 \
    --model mistral-7b-instruct
```

### Verifica Connessione

```bash
# Test connessione Ollama
curl http://localhost:11434/api/tags

# Oppure da Python
python -c "from experiments_llm.backends.ollama_backend import OllamaBackend; \
           b = OllamaBackend('test'); \
           print('Ollama:', 'OK' if b.is_available() else 'FAIL')"
```

---

## 📈 Performance Attese

### Hardware Minimo

- **CPU**: 4 cores, 8 threads
- **RAM**: 16GB
- **Disco**: 10GB per modelli

### Hardware Raccomandato

- **CPU**: 8+ cores o GPU (NVIDIA RTX 3060+)
- **RAM**: 32GB+
- **Disco**: 50GB per tutti i modelli

### Tempi Stimati

| Configurazione | Fast Mode | Full Mode |
|----------------|-----------|-----------|
| Mistral 7B (CPU) | 2-3h | 6-8h |
| Mistral 7B (GPU) | 1-1.5h | 3-4h |
| Llama 3.2 3B (CPU) | 1.5-2h | 4-5h |
| GPT-OSS 20B (GPU) | 4-5h | 12-15h |

---

## 🔬 Metodologia

### Differenze Simulazione vs LLM

**Simulazione Originale:**
```python
# Decisione deterministica
if roi > 1.0:
    return COOPERATE
```

**LLM Reale:**
```python
prompt = f"Your cooperation ROI is {roi:.2f}. Cooperate or defect?"
response = llm.generate(prompt)
# Parse: "COOPERATE" or "DEFECT"
```

### Vantaggi LLM

✅ Comportamento emergente (non programmato)
✅ Ragionamento linguistico naturale
✅ Valida leggi NYX su AI reali

### Sfide LLM

⚠️ Bias da RLHF (troppo "helpful")
⚠️ Variabilità stocastica
⚠️ 10-100x più lento

---

## 🎯 Modelli Raccomandati

### Per Ricerca (minimo bias)

1. **Mistral 7B Base** (se disponibile, non instruct)
2. **Llama 3.2 Base** (non instruct)

### Per Esperimenti Pratici

1. **Mistral 7B Instruct** ⭐ (miglior equilibrio)
2. **Llama 3.2 3B Instruct** (veloce)
3. **Phi 3.5 Mini** (molto veloce, ma più biased)

### Evitare

❌ **GPT-4o/Claude** (troppo RLHF, API a pagamento)
❌ **DialogPT** (progettato per dialogo, non decision-making)

---

## 📝 Note Metodologiche

### Bias RLHF

Tutti i modelli instruct hanno bias verso cooperazione. Per compensare:

- Usare `temperature > 0.7` per più variabilità
- Includere prompt adversarial
- Documentare bias nei risultati

### Riproducibilità

- Seed non garantisce risultati identici (LLM stocastici)
- Eseguire multiple run e fare media
- Salvare prompt esatti usati

---

## 🐛 Troubleshooting

### Ollama non risponde

```bash
# Windows: Riavvia app Ollama
# Linux:
sudo systemctl restart ollama

# Verifica porta
netstat -an | grep 11434
```

### Modello non trovato

```bash
ollama list  # Verifica modelli installati
ollama pull mistral-7b-instruct  # Scarica se mancante
```

### Out of Memory

```bash
# Usa modello più piccolo
python experiments_llm/run_historical_reconstruction.py \
    --model phi-3.5-mini \
    --fast
```

### LLM responses troppo lenti

```bash
# Se su CPU, riduci max_tokens
# Modifica in backends/ollama_backend.py:
# max_tokens = 20 → 10
```

---

## 📚 Riferimenti

- **Paper NYX**: arXiv:2024.XXXXX
- **Ollama Docs**: https://github.com/ollama/ollama
- **Repository Originale**: [link al repo]

---

## 🤝 Contributi

Per contribuire con nuovi backend (HuggingFace, vLLM, etc.):

1. Implementa `BaseLLMBackend` in `backends/`
2. Testa con `test_models_bias.py`
3. Apri PR con risultati

---

**Buona ricostruzione scientifica! 🔬🚀**
