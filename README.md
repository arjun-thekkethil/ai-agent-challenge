# Agent-as-Coder — Bank Statement Parser Agent

An autonomous agent that generates, tests, and refines Python parsers to extract transaction data from bank statement PDFs into `pandas.DataFrame` format — fully matching a reference CSV schema.  
Supports **self-fixing** loops (≤3 attempts) and uses the **Google Gemini API** (free tier) by default, with optional Groq support.

---

## 5-Step Run Instructions

### **1️⃣ Fork and Clone the Repository**
```bash
git clone https://github.com/<your-username>/ai-agent-challenge.git
cd ai-agent-challenge
git checkout -b feature/agent

2️⃣ Create & Activate Virtual Environment
python -m venv venv
venv/Scripts/python.exe -m pip install --upgrade pip
source venv/Scripts/activate
3️⃣ Install Dependencies
venv/Scripts/python.exe -m pip install -r requirements.txt
4️⃣ Set API Key(s)
Obtain a free Gemini API key from Google AI Studio.
export GEMINI_API_KEY="your_gemini_key_here"
export LLM_PROVIDER="gemini"
5️⃣ Run the Agent for a Target
Example: ICICI Bank
python agent.py --target icici

This will:

Read the bank’s sample PDF & reference CSV from data/<bank_name>/

Generate custom_parsers/<bank_name>_parser.py

Test the parser output against the CSV (DataFrame.equals)

Auto-fix up to 3 times until matched

📊 Agent Workflow
Read sample PDF & CSV

Prompt LLM (Gemini or Groq) to generate parse(pdf_path) -> pandas.DataFrame

Import & run parser

Compare output with CSV using DataFrame.equals

Auto-fix on failure, up to 3 attempts

📂 Project Structure
ai-agent-challenge/
├── agent.py                  # Main agent script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── .gitignore
├── data/
│   ├── icici/
│   │   ├── icici_sample.pdf
│   │   └── icici_sample.csv
│   └── hdfc/
│       ├── hdfc_sample.pdf
│       └── hdfc_sample.csv
└── custom_parsers/           # Generated parser files

🧪 Example Run Output
Using provider: gemini
PDF: data\icici\icici_sample.pdf
CSV: data\icici\icici_sample.csv

=== Attempt 1/3 ===
Test failed (DataFrame mismatch)...
=== Attempt 2/3 ===
SUCCESS — parser produced DataFrame matching CSV.
DataFrames match exactly (equals=True)

⚠️ Limitations
Assumes consistent PDF format per bank.

No OCR — works only with extractable text PDFs.

Internet connection required for LLM calls.

Maximum 3 auto-fix attempts per run.

📜 License
This project is for the Karbon AI Agent Challenge.
Educational/demo purposes only.