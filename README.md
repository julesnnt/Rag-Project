# RAG Project — NBA Player Q&A System (RAPTOR Statistics)

**Deep Learning — Bsc Data Science for Responsible Business — Centrale Lyon 2025-2026**

---

## Objective

This project implements a complete **Retrieval-Augmented Generation (RAG)** system capable of answering natural language questions about advanced NBA player statistics. The pipeline combines state-of-the-art Hugging Face Large Language Models with ChromaDB, an open-source vector database, to create an intelligent question-answering system that retrieves relevant information and generates accurate, contextual answers.

---

## Dataset

**`538_historical_RAPTOR_by_player.csv`** — FiveThirtyEight  
Advanced NBA statistics by player and season from 1977 onwards.

| Column | Description |
|--------|-------------|
| `player_name` | Player name |
| `season` | NBA season |
| `raptor_offense` | Offensive RAPTOR score |
| `raptor_defense` | Defensive RAPTOR score |
| `raptor_total` | Total RAPTOR score |
| `war_total` | Wins Above Replacement (total) |
| `predator_total` | PREDATOR predictive score |
| `pace_impact` | Impact on game pace |
| `mp` | Minutes played |
| `poss` | Number of possessions |

---

## RAG Pipeline Architecture

```
         ┌─────────────┐
  Query  │             │  Embedding (Sentence Transformer)
 ───────▶│   Encode    │──────────────────────────────────┐
         │             │                                  │
         └─────────────┘                                  ▼
                                                 ┌─────────────────┐
                                                 │    ChromaDB     │
         ┌─────────────┐                         │  Vector Search  │
         │  Response   │◀── LLM Generation ──────│   (Top-k docs)  │
         └─────────────┘        ▲                └─────────────────┘
                                │                        │
                         ┌──────┴──────┐                 │
                         │   Prompt    │◀────────────────┘
                         │  Template   │   Context
                         └─────────────┘
```

---

## Technical Components

### 1. Data Preparation
- Loading and exploring the RAPTOR dataset (15+ columns, thousands of rows)
- Cleaning missing values and preprocessing
- Converting tabular data into structured text documents (one document per player/season)

### 2. Embedding Generation
Two Hugging Face models compared:

| Model | Dimensions | Speed | Use Case |
|-------|-----------|-------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Production |
| `all-mpnet-base-v2` | 768 | Moderate | Maximum quality |

### 3. Vector Database — ChromaDB
- Local persistent deployment (`./raptor_chroma_db/`)
- 2 indexed collections (one per embedding model)
- Cosine similarity metric (HNSW index)
- Batch indexing for large volumes

### 4. Hugging Face LLM
- **Main model**: `google/flan-t5-base` (Seq2Seq, lightweight, no GPU required)
- GPU (CUDA) compatible when available

### 5. Prompt Engineering
4 templates tested and compared:

| Template | Description |
|----------|-------------|
| `standard` | XML format with `<context>` and `<question>` tags |
| `concise` | Short, direct answers |
| `expert` | NBA analytics expert role |
| `chain_of_thought` | Step-by-step reasoning |

### 6. End-to-End Pipeline
```
Query → Embedding → ChromaDB Retrieval → Prompt Construction → LLM Generation → Response
```

### 7. Gradio Interface
Conversational web interface featuring:
- Prompt template selection
- Adjustable k parameter (number of retrieved documents)
- Source display and latency metrics

---

## Evaluation & Results

### Embedding Model Comparison

| Metric | MiniLM-L6-v2 | MPNet-base-v2 |
|--------|-------------|--------------|
| Hit Rate @3 | empirically compared | empirically compared |
| Retrieval latency | faster | slower |
| Vector dimension | 384 | 768 |

### Evaluated Metrics
- **Hit Rate @k**: proportion of queries where the correct document appears in the top-k results
- **Latency**: retrieval time, generation time, and total response time
- **Response quality**: relevance and accuracy of generated answers
- **Impact of k**: Hit Rate evolution as the number of retrieved documents varies

---

## Installation

```bash
pip install transformers chromadb sentence-transformers huggingface-hub \
            langchain_community langchain-text-splitters pypdf \
            gradio tqdm pandas matplotlib seaborn scikit-learn accelerate
```

---

## Usage

Open and execute the notebook cells in order:

```bash
jupyter notebook RAG_RAPTOR_NBA.ipynb
```

**Notebook sections:**
1. Installation & Imports
2. Dataset Exploration
3. Document Preparation
4. Embedding Generation
5. ChromaDB Vector Store
6. Hugging Face LLM Integration
7. Complete RAG Pipeline
8. Evaluation & Optimization
9. Gradio Interface

---

## Project Structure

```
Project Rag/
├── README.md
├── RAG_RAPTOR_NBA.ipynb                    # Main notebook
├── 538_historical_RAPTOR_by_player.csv     # Dataset
└── raptor_chroma_db/                       # ChromaDB (created at runtime)
```

---

## Example Queries

- *Who had the highest RAPTOR total in the 1996 NBA season?*
- *What is Michael Jordan's offensive RAPTOR in 1991?*
- *Which player had the best WAR total ever?*
- *Who was the best defensive player in 2004?*
- *LeBron James statistics in 2013*

---

## Expected Outcomes

- A fully functional RAG system capable of answering user queries by leveraging an external knowledge base
- Proficiency in using ChromaDB for semantic search tasks
- Practical experience deploying Hugging Face LLMs in real-world scenarios
- Critical analysis of system performance metrics: accuracy, relevance, response time, and scalability

---

## Author

**Course**: Deep Learning — Emmanuel Dellandréa  
**Program**: Bsc Data Science for Responsible Business — Centrale Lyon  
**Academic Year**: 2025-2026  
**Deadline**: Tuesday, April 21, 2026
