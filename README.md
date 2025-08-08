# SEC Filings QA Agent

A question-answering system over SEC filings. It fetches company filings from EDGAR, processes and stores them in vector format in CockroachDB, and answers natural language questions using Google Gemini (Generative AI).

---

## 📚 Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Demo](#running-the-demo)
- [Using the System in Your Code](#using-the-system-in-your-code)
- [Project Structure](#project-structure)

---

## ✅ Prerequisites

1. **Python (via Conda or pip)**  
   Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended), or use pip with Python 3.11+.

2. **CockroachDB**  
   - Sign up: [cockroachlabs.cloud](https://cockroachlabs.cloud/signup)  
   - Create a free cluster and copy your connection string.

3. **Google Gemini API Key**  
   - Enable the Generative AI API and obtain your key:  
     [Google AI Studio](https://aistudio.google.com/app/apikey)

4. **EDGAR Tools Library**  
   - [Documentation](https://edgartools.readthedocs.io/en/latest/)

---

## ⚙️ Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/AxelB1011/ScalarField_Screening.git
   cd ScalarField_Screening
   ```

2. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate scalarfield
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root with the following:
   ```env
   EDGAR_IDENTITY="Your Name your.email@domain.com"
   GEMINI_API_KEY="YOUR_GOOGLE_GENAI_API_KEY"
   COCKROACH_DATABASE_URL="YOUR_COCKROACH_DB_URL"
   ```
   > ⚠️ Replace `sslmode=verify-full` with `sslmode=require` in the Cockroach URL if needed.

---

## 🚀 Running the Demo

Run the demo script to ingest filings and test Q&A:
```bash
python demo-clean.py
```

Sample questions:
- What was Apple’s revenue in the latest quarter?
- How much does Apple spend on R&D?
- Compare R&D spending trends across companies. What insights about innovation investment strategies?

---

## 🛠️ Using the System in Your Code

You can use the `SECQASystem` class in your own scripts:

```python
import asyncio
from pipeline import SECQASystem

async def run_query():
    system = SECQASystem()
    await system.initialize()
    await system.ingest_companies(["MSFT"], limit_per_company=3)
    result = await system.ask_question("What are Microsoft’s risk factors?", max_sources=5)
    print(result.answer)
    print(result.sources)

asyncio.run(run_query())
```
```python3
# Modifiable variables/parameters in demo-clean.py

print("\n📊 Ingesting sample companies...")
companies = ["AAPL", "MSFT", "TSLA", "JPM"]
results = await system.ingest_companies(companies, limit_per_company=3)

print("\n❓ Demo questions:")
test_cases = [
  "What was Apple's revenue in the latest quarter?", 
  "How much does Apple spend on R&D?",
  "Compare R&D spending trends across companies. What insights about innovation investment strategies?",
  "Analyze recent executive compensation changes. What trends emerge?",
]




```

---

## 🗂️ Project Structure

| File/Module        | Description                                         |
|--------------------|-----------------------------------------------------|
| `chunking.py`      | SEC form parsers and chunk creation logic           |
| `routing.py`       | Concept routing and query mapping                   |
| `storage.py`       | Vector DB schema, ingestion, and search using CockroachDB |
| `pipeline.py`      | Main QA pipeline engine                             |
| `demo-clean.py`    | CLI demo for ingestion and Q&A                      |
| `utilities.py`     | Helpers for CIK/ticker mapping, date parsing, etc.  |
| `environment.yml`  | Conda environment specification                     |

---

## 📬 Contact

For bugs or feature requests, open an issue or submit a PR or email gopalk045@gmail.com

---

## Sample Outputs


Sample Terminal Output:
```python3
❓ Demo questions:

🤔 What was Apple's revenue in the latest quarter?
INFO:pipeline:Processing question: What was Apple's revenue in the latest quarter?
INFO:pipeline:Routing: Identified primary concept: Financial Statements. Temporal strategy: recent. Best matches: 10-K ITEM 8, 8-K ITEM 2.02, 10-Q ITEM 1
INFO:pipeline:Found 5 candidate chunks
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cpu
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: intfloat/e5-small-v2
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:google_genai.models:AFC remote call 1 is done.
INFO:pipeline:Generated answer using gemini in 18.5s
 📈 Confidence: 0.97
 🎯 Routing: Identified primary concept: Financial Statements. Temporal strategy: recent. Best matches: 10-K ITEM 8, 8-K ITEM 2.02, 10-Q ITEM 1
 📚 Sources: (5) sources: [[1, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [2, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000071/aapl-20250731.htm'], [3, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000071/aapl-20250731.htm'], [4, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm'], [5, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm']]
 🤖 Provider: gemini
 ⏱️ Time: 18.5s
 💬 Answer: **Key Findings:**

Apple Inc.'s total net sales for its second fiscal quarter ended March 29, 2025, were **$95,359 million**. This represents a 5% increase compared to the $90,753 million in total net sales reported for the same period in the prior fiscal year (three months ended March 30, 2024) [1]...
   📋 Top sources:
     - AAPL 10-Q - A of the 2024 Form 10-K and Part II, Item 1A of th...
     - AAPL 8-K - Results of Operations and Financial Condition....
     - AAPL 8-K - Financial Statements and Exhibits....

🤔 How much does Apple spend on R&D?
INFO:pipeline:Processing question: How much does Apple spend on R&D?
INFO:pipeline:Routing: Identified primary concept: Research Development. Temporal strategy: recent. Best matches: 10-K ITEM 7, 10-K ITEM 1, 10-Q ITEM 1
INFO:pipeline:Found 8 candidate chunks
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:google_genai.models:AFC remote call 1 is done.
INFO:pipeline:Generated answer using gemini in 7.6s
 📈 Confidence: 0.97
 🎯 Routing: Identified primary concept: Research Development. Temporal strategy: recent. Best matches: 10-K ITEM 7, 10-K ITEM 1, 10-Q ITEM 1
 📚 Sources: (5) sources: [[1, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [2, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000071/aapl-20250731.htm'], [3, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000071/aapl-20250731.htm'], [4, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm'], [5, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm']]
 🤖 Provider: gemini
 ⏱️ Time: 7.6s
 💬 Answer: **Key Findings:**

The provided excerpts from Apple Inc.'s (AAPL) FY2025 Q2 Form 10-Q [1] do not contain the "Consolidated Statements of Operations," which is the financial statement where Research and Development (R&D) expenses are typically reported. Therefore, the precise quantitative amount Appl...
   📋 Top sources:
     - AAPL 10-Q - A of the 2024 Form 10-K and Part II, Item 1A of th...
     - AAPL 8-K - Financial Statements and Exhibits....
     - AAPL 8-K - Results of Operations and Financial Condition....

🤔 Compare R&D spending trends across companies. What insights about innovation investment strategies?
INFO:pipeline:Processing question: Compare R&D spending trends across companies. What insights about innovation investment strategies?
INFO:pipeline:Routing: Identified primary concept: Research Development. Temporal strategy: historical. Best matches: 10-K ITEM 7, 10-K ITEM 1, 10-Q ITEM 1
INFO:pipeline:Found 6 candidate chunks
INFO:google_genai.models:AFC is enabled with max remote calls: 10.
INFO:httpx:HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
INFO:google_genai.models:AFC remote call 1 is done.
INFO:pipeline:Generated answer using gemini in 8.6s
 📈 Confidence: 0.97
 🎯 Routing: Identified primary concept: Research Development. Temporal strategy: historical. Best matches: 10-K ITEM 7, 10-K ITEM 1, 10-Q ITEM 1
 📚 Sources: (5) sources: [[1, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [2, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm'], [3, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm'], [4, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [5, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000008/aapl-20241228.htm']]
 🤖 Provider: gemini
 ⏱️ Time: 8.6s
 💬 Answer: **Key Findings:**

The provided excerpts from Apple Inc.'s (AAPL) Form 10-Q filings for fiscal year 2025 Q2, 2025 Q3, and 2024 Q1 **do not contain the Consolidated Statements of Operations**. Research and Development (R&D) expenses are typically reported as a line item within the operating expenses ...
   📋 Top sources:
     - AAPL 10-Q - A of the 2024 Form 10-K and Part II, Item 1A of th...
     - AAPL 10-Q - Item 1.     Financial Statements                  ...
     - AAPL 10-Q - Item 1.     Financial Statements                  ...

```

Sample Notebook Output:
```python3
🚀 SEC QA System initialized successfully!

📊 Ingesting sample companies...
  ✅ AAPL: 84 chunks in 28.1s

❓ Demo questions:

🤔 What are Apple's main risk factors?
   📈 Confidence: 0.97
   🎯 Routing: Analysis focused on: AAPL. Temporal strategy: Prioritize most recent filings. Primary sections: 8-K ITEM 8.01, 10-Q ITEM 1A, 10-K ITEM 1A
   📚 Sources: (5) sources: [[1, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm'], [2, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [3, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [4, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000114036125027340/ef20052355_8k.htm'], [5, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000114036125027340/ef20052355_8k.htm']]
   🤖 Provider: gemini
   ⏱️ Time: 15.7s
   💬 Answer: The provided context indicates that a comprehensive discussion of Apple's risk factors is located in Part I, Item 1A of the Annual Report on Form 10-K for the fiscal year ended September 28, 2024 (the "2024 Form 10-K") and Part II, Item 1A of the Quarterly Report on Form 10-Q [1, Part I, Item 2; 2, Part I, Item 2]. Since these specific sections containing the full list of risk factors are not fully provided in the context, a complete list cannot be given.

However, based on the "Management’s Discussion and Analysis of Financial Condition and Results of Operations" sections of the FY2025 Q3 and Q2 Form 10-Q filings, the following factors are highlighted as having directly and indirectly impacted, or potentially materially impacting, the Company’s results of operations and financial condition:

*   **Macroeconomic Conditions:** These include inflation, interest rates, and currency fluctuations [1, Part I, Item 2, Macroeconomic Conditions; 2, Part I, Item 2, Macroeconomic Conditions].
*   **Tariffs and Other Measures:** This encompasses new U.S. Tariffs, additional tariffs on imports from various countries (such as China, India, Japan, South Korea, Taiwan, Vietnam, and the European Union), and reciprocal tariffs or retaliatory measures imposed by other countries. These measures can materially adversely affect Apple's supply chain, the availability of rare earths and other raw materials and components, pricing, and gross margin. Furthermore, trade and other international disputes can negatively impact the overall macroeconomic environment, potentially leading to shifts and reductions in consumer spending and negative consumer sentiment for Apple's products and services [1, Part I, Item 2, Tariffs and Other Measures; 2, Part I, Item 2, Tariffs and Other Trade Measures].
*   **Business Seasonality and Product Introductions:** The Company historically experiences higher net sales in its first fiscal quarter due to seasonal holiday demand. The timing of new product and service introductions can significantly impact net sales, cost of sales, and operating expenses. Net sales can also be affected when consumers and distributors anticipate a new product launch [1, Part I, Item 2, Business Seasonality and Product Introductions; 2, Part I, Item 2, Business Seasonality and Product Introductions]....

🤔 What is the trend in Apple's recent revenue ?
   📈 Confidence: 0.97
   🎯 Routing: Analysis focused on: AAPL. Temporal strategy: Analyze across multiple periods for trend analysis. Primary sections: 8-K ITEM 2.02, 10-Q ITEM 2, 10-Q ITEM 1
   📚 Sources: (5) sources: [[1, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000073/aapl-20250628.htm'], [2, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [3, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000057/aapl-20250329.htm'], [4, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000071/aapl-20250731.htm'], [5, 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000114036125027340/ef20052355_8k.htm']]
   🤖 Provider: gemini
   ⏱️ Time: 17.2s
   💬 Answer: Apple's recent revenue shows an increasing trend.

For the three months ended June 28, 2025 (Q3 FY2025), Apple reported total net sales of **$94,036 million**, an increase of 10% compared to **$85,777 million** for the same period in 2024 [1, p. 1, 14]. This increase was primarily driven by higher net sales of Products, which rose to $66,613 million from $61,564 million, and Services, which increased to $27,423 million from $24,213 million [1, p. 1]. Specifically, iPhone net sales increased by 13% to $44,582 million, Mac net sales increased by 15% to $8,046 million, and Services net sales increased by 13% to $27,423 million [1, p. 15].

For the first nine months of fiscal year 2025 (ending June 28, 2025), total net sales were **$313,695 million**, up 6% from **$296,105 million** in the first nine months of fiscal year 2024 [1, p. 1, 14]. Products sales for this period increased to $233,287 million from $224,908 million, and Services sales increased to $80,408 million from $71,197 million [1, p. 1].

This positive trend is also evident in the prior quarter. For the three months ended March 29, 2025 (Q2 FY2025), total net sales increased by 5% to **$95,359 million** from **$90,753 million** in Q2 FY2024 [3, p. 1, 2, p. 13]. Product sales for Q2 FY2025 were $68,714 million (up from $66,886 million) and Services sales were $26,645 million (up from $23,867 million) [3, p. 1].

For the first six months of fiscal year 2025 (ending March 29, 2025), total net sales were **$219,659 million**, an increase of 4% from **$210,328 million** in the first six months of fiscal year 2024 [3, p. 1, 2, p. 13]....

📊 System Status:
   status: healthy
   database_provider: cockroach
   chunks_stored: 2229
   companies_indexed: 4
   cache_stats: {'ticker_cache_size': 0, 'embedding_cache_size': 82, 'query_cache_size': 3}

✅ Demo complete!
```