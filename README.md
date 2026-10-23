# Python Tutorial Q&A

This is a small question answering service over the official Python 3 Tutorial. You send a question and it sends back an answer plus the passages it used. If the tutorial does not have the answer, the service says so and does not guess.

I made this for the Hack On Tech AI Engineering Internship task. I built it step by step in Google Colab with help from an AI assistant (Claude), and I ran and tested every part myself.

## What is in this repo

| File | What it does |
|---|---|
| `scraper.py` | Downloads the tutorial and splits it into sections |
| `index.py` | Makes chunks, builds the search index, and has the search code |
| `app.py` | The FastAPI service with the `/ask` endpoint |
| `evaluate.py` | Calculates Hit Rate@k for the questions |
| `questions.json` | My 20 test questions with the correct section for each |
| `questions_extra.json` | 2 extra questions I used after finding failures |
| `test_app.py`, `try_questions.py`, `check_refusals.py`, `ask_one.py` | Small scripts to try the full service |
| `data/` | Scraped sections, chunks, and embeddings (already built) |
| `results/` | `results.csv` with every evaluation run, and the saved answers from my test runs |

## How to run it

You need Python 3.10 or higher, git, and a free Groq API key (get one at console.groq.com).

**Step 1. Clone the repo**

```bash
git clone https://github.com/Alishba-12/python-docs-qa.git
cd python-docs-qa
```

**Step 2. Make a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

On Windows use `venv\Scripts\activate` for the second line. You should see `(venv)` at the start of your terminal line.

**Step 3. Install the packages**

```bash
pip install -r requirements.txt
```

The first time you run the service it also downloads a small embedding model, so it needs internet and a minute or two.

**Step 4. Add your Groq key**

```bash
cp .env.example .env
```

On Windows use `copy .env.example .env`. Open the new `.env` file and replace `your_groq_key_here` with your real key. This file is ignored by git, so your key stays private.

Groq sometimes removes models. The service uses `openai/gpt-oss-120b` by default. If you get a "model not found" error, add a line like `GROQ_MODEL=another-model-name` to your `.env` file.

**Step 5. Start the service**

```bash
uvicorn app:app --port 8000
```

Wait until you see `Application startup complete`. The first start takes a few seconds because it loads the index.

**Step 6. Ask a question**

The easiest way is to open `http://127.0.0.1:8000/docs` in your browser, click `POST /ask`, click "Try it out", type a question, and press Execute.

You can also use curl:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I write an if statement?"}'
```

The response has three parts:

```json
{
  "answer": "...",
  "grounded": true,
  "sources": [
    {"section_id": "controlflow.html#if-statements", "url": "...", "title": "...", "text": "...", "score": 0.03}
  ]
}
```

If the tutorial does not cover the question, `grounded` is `false`, `sources` is empty, and the answer says the documentation does not contain it.

**Step 7 (optional). Check the Hit Rate**

```bash
python evaluate.py
```

This runs my 20 questions and prints Hit Rate@5 and Hit Rate@1. You can also try `--mode vector`, `--mode bm25`, or `--k 10`. Every run is added to `results/results.csv`.

**Step 8 (optional). Rebuild everything from scratch**

```bash
python scraper.py
python index.py
```

The data folder is already in the repo, so you only need this if you want to rebuild it.

## How it works

1. **Scraping.** `scraper.py` reads the tutorial table of contents and splits every page by its headings. Each section gets an id like `controlflow.html#if-statements`. This gave me 132 sections.
2. **Chunking.** A short section stays as one chunk. A long section is cut into pieces of 150 words with 30 words of overlap, so an answer that sits on a boundary still appears whole in one chunk. This gave 338 chunks. I also add the section title to each chunk for searching.
3. **Search.** I use two searches together. One is embedding search with `all-MiniLM-L6-v2`, which finds chunks with a similar meaning. The other is BM25, which finds exact words. I merge the two rankings with Reciprocal Rank Fusion (top 50 from each list, constant 60). The top 10 chunks go to the LLM.
4. **Not guessing.** There are two layers. First, if the best cosine score among the retrieved chunks is below 0.25, the service refuses without calling the LLM. Second, the LLM is told to use only the passages and to reply `NOT_IN_DOCS` if they do not directly answer the question. I picked 0.25 from a handful of test questions, so it is a rough guess.
5. **LLM.** Groq with `openai/gpt-oss-120b` and temperature 0. I chose it because Groq has a free tier.
6. **Sources.** The service returns only the passages that the answer cites, like `[2]`. If the answer cites nothing, it returns all of them.

## Hit Rate@5 results

My test set is `questions.json`, with 20 questions. Each one has the correct section. A question counts as a hit if a chunk from the correct section is in the top 5 results.

| Setup | Hit Rate@5 | Hit Rate@1 |
|---|---|---|
| **Baseline** (hybrid, 150-word chunks, MiniLM) | **19/20 (95%)** | 70% |
| Vector search only | 18/20 (90%) | 85% |
| Keyword search (BM25) only | 18/20 (90%) | 70% |
| Chunks of 75 words | 18/20 (90%) | 65% |
| Chunks of 300 words | 18/20 (90%) | 65% |
| `bge-small-en-v1.5` embeddings | 19/20 (95%) | 80% |

**What I tried and what happened**

- **Search method.** Hybrid got 95% and each single search got 90%, so I kept hybrid.
- **Chunk size.** Both 75 and 300 words scored one question lower than 150 words, so I kept 150.
- **Embedding model.** `bge-small` matched the baseline on Hit Rate@5 and put the right section first more often (80% against 70%). I did not switch because my 0.25 score gate was picked from MiniLM scores, and bge scores sit in a different range. I also did not add the query instruction that bge works best with, so it may be a bit underrated here.
- **Passages sent to the LLM (5 or 10).** After I found failures, I tested the walrus operator question and the timing code question. With 5 passages hybrid found 0 of 2, and with 10 it found 2 of 2. Vector alone and BM25 alone each found 1 of 2. On my 20 questions, going from 5 to 10 stayed at 19/20, so it did not hurt. I changed the service to use 10 passages.

**Please read this part.** My test is small, and one question is worth 5 points, so a difference of one question is not strong proof. Most of my questions are reworded section titles, so they are fairly easy. The two extra questions were picked after I saw the service fail on them, so they are examples and not a fair test.

## Refusal checks

I also tried 10 questions that I believe the tutorial does not cover (for example the capital of France, PyTorch, asyncio, decorators, MySQL, the GIL, and reading or writing CSV files). The service refused all 10. For some of them the best cosine score was above 0.25 (for example 0.48 for sorting a dictionary by its values and 0.39 for writing CSV rows), so the second layer, the LLM itself, did the refusing there. This is a small check and not a proper test.

## Three questions my service gets wrong

I saved the answers from three runs in `results/`: `qa_log_before.json` (first version), `qa_log_prompt_v2.json` (stricter prompt), and `qa_log.json` (10 passages).

**1. "How do I reverse a string?"**

The service gives loops that print characters backwards (an iterator class, a generator, and a `reversed('abc')` example). None of them gives back a reversed string. The tutorial only shows `msg[::-1]` inside an example about `from sound.effects import *`, so the docs have no direct answer. The service should have said so. I also could not find the `reversed('abc')` example anywhere in the tutorial text (the only `reversed(` example I found is `reversed(range(1, 10, 2))`), so I think the model wrote it itself, yet it added a citation to it.

*Why:* the word "reverse" matched the iterators section strongly (cosine 0.58, the highest I saw), and my score gate cannot catch that, because it measures how close the topic is and not whether the passage answers the question. A stricter prompt and more passages did not change it.

**2. "How do I make a variable private inside a class?"**

The service finds the right section, but the answer is misleading. The tutorial says three things in order: truly private variables do not exist in Python, a name with one leading underscore is the convention for non-public names, and name mangling (`__spam`) is limited support that exists to avoid name clashes with subclasses. The latest answer calls the double underscore "the convention" and "the supported way to hide a name", and it leaves out the single underscore.

*Why:* I am not fully sure. The answer was better in my stricter-prompt run with 5 passages (it mentioned both the single underscore and that true private variables do not exist), and it got worse after I sent 10 passages. I only ran each version once, so I cannot say the extra passages caused it.

**3. "Where does Python look for a module when I import it?"**

This is the one question my retrieval check misses (the correct section is `modules.html#the-module-search-path`). The service does answer it, but the answer is thin: "Python searches the directories listed on `sys.path` to locate the module (or package) you import." That is true, but it comes from the Packages section (`modules.html#packages`) and not from the right section. The right section also says that Python checks built-in modules first, then the directory of the input script, then `PYTHONPATH`, then the installation default directories. None of that is in the answer. So the answer is not wrong, but it is incomplete and it cites the wrong section.

*Why:* the section talks about `sys.path` and `PYTHONPATH`, but my question says "look for" and "import", so the keyword search never found it. Vector search ranked it 1st, keyword search did not have it in its top 50, and the merged ranking put it 18th. The service only sends 10 chunks to the LLM, so the right section never reached it, and the model answered from a passage that only mentions `sys.path` in passing.

This is the same pattern as the walrus question (keyword search 1st, vector search 57th) and the timing question (vector search 1st, keyword search 58th). In my merge, a chunk that only one search finds earns at most 1/61 (about 0.016), while a chunk ranked around 10th in both lists earns about 0.029. So chunks that are only a little relevant in both lists beat a chunk that one search ranks first. The walrus and timing questions got fixed only because their merged rank was 9, which just fits in 10 passages.

## What I would improve next

1. **Fix the merge.** Always keep the top 3 results from each search before filling the rest with the merged ranking. By the rank numbers above, this should rescue all three retrieval failures. I did not test it because of time.
2. **Bigger and harder test set.** 20 questions is too small. I would write questions that do not copy section titles, and add questions the docs do not cover, so refusals are measured too.
3. **Check answers, not only retrieval.** Right now Hit Rate only tells me if the right section was found. I would also measure if the final answers are correct and if the refusals are right.
4. **Stronger guard against guessing.** I would ask the model to quote the exact sentence it used and refuse if that sentence is not in the passages. This should catch the reverse string case.
5. **Tune the score gate and try bge properly.** Use the query instruction for bge, set the 0.25 gate again, and rerun the comparison.
6. **Cleaner sources.** Remove repeated chunks from the same section in the sources list.
