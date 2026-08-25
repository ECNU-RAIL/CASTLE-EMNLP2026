# CASTLE: A Comprehensive Benchmark for Evaluating Student-Tailored Personalized Safety in Large Language Models

> Official implementation of the **EMNLP 2026** paper
> **“CASTLE: A Comprehensive Benchmark for Evaluating Student-Tailored Personalized Safety in Large Language Models”**

CASTLE evaluates personalized safety in educational LLM interactions. It covers four educational risk categories, 15 risk domains, 14 student attributes, and 92,908 bilingual Chinese--English scenarios.

## Repository Structure

```text
.
├── data/
│   ├── CASTLE_zh.json       # 53,483 Chinese scenarios
│   └── CASTLE_en.json       # 39,425 English scenarios
├── src/
│   ├── generate_profiles.py  # Seed-based student-profile construction
│   ├── generate_queries.py   # Dual-prompt query construction
│   ├── collect_responses.py # Generate non-personalized and personalized responses
│   ├── score_responses.py   # Evaluate responses with three CASTLE dimensions
│   ├── summarize_results.py # Aggregate scores by model and risk domain
│   └── castle_benchmark/    # Shared data, prompt, API, and result utilities
└── requirements.txt
```

## Installation

```bash
git clone <repository-url>
cd castle
pip install -r requirements.txt
```

## Environment Setup

The scripts use an OpenAI-compatible API. Keep credentials in environment variables; no API key or private endpoint is included in this repository.

```bash
export OPENAI_API_KEY=YOUR_KEY
# Optional for an OpenAI-compatible provider:
export OPENAI_BASE_URL=https://api.openai.com/v1
```

Alternatively, place the same variables in a local `.env` file. It is excluded from version control.

## Full Evaluation Pipeline

### 0. Optional dataset construction

The released `data/` files are the final benchmark. To construct a new benchmark split, first expand existing seed profiles according to Appendix A, then generate queries. Both scripts write the current CASTLE schema only: scenario type, scenario subtype, profile, and (after query generation) query.

```bash
python src/generate_profiles.py \
  --seed-file path/to/seed_profiles_en.json \
  --output generated_profiles_en.json \
  --language en \
  --models model-a,model-b

python src/generate_queries.py \
  --input generated_profiles_en.json \
  --output generated_scenarios_en.json \
  --language en \
  --models model-a,model-b
```

Use `--dry-run` before API calls. Profile generation follows the seed-based construction, educational-psychology frameworks, structural schema, and consistency constraints in Appendix A. Query generation uses the original short/rich prompt rotation.

### 1. Generate model responses

For each scenario, the script generates a non-personalized response using only the query and a personalized response using the query plus the student profile.

```bash
python src/collect_responses.py \
  --input data/CASTLE_en.json \
  --language en \
  --models your-model-id \
  --workers 4
```

For the Chinese benchmark, replace the input and language with `data/CASTLE_zh.json` and `zh`.

### 2. Score the responses

Responses are scored from 1 to 5 on Risk Sensitivity, Emotional Empathy, and User-specific Alignment. The total score is the sum of the three dimensions (out of 15).

```bash
python src/score_responses.py \
  --responses results/responses \
  --judge your-judge-model \
  --workers 4
```

### 3. Summarize results

```bash
python src/summarize_results.py \
  --scores results/scores \
  --output-dir results/summary
```

The script writes CSV and JSON summaries for each response model, judge, language, condition, risk category, and risk domain.

## Dataset

The two JSON files are UTF-8 arrays. Each Chinese record contains `场景类型`, `场景子类型`, `用户查询`, and `用户画像`. Each English record contains `Scenario_Type`, `Scenario_Subtype`, `User_Query`, and `User_Profile`.

The dataset covers the following categories:

- Psychological and Emotional Health
- Academic Integrity and Competence
- Content and Information Bias
- Learning Dependence and Cognition

## Prompt Protocol

The response-generation prompts and LLM-as-a-judge prompts in this repository are retained from the original CASTLE experimental code without wording changes.

## Safety Note

CASTLE is a research benchmark containing high-risk educational scenarios. It must not be used as clinical advice, crisis intervention, or the sole basis for decisions about real students.
