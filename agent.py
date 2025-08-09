"""
Agent-as-Coder: agent.py
Usage:
    python agent.py --target icici
Produces:
    custom_parsers/<target>_parser.py
Contract:
    - Generated module must implement parse(pdf_path: str) -> pandas.DataFrame
    - agent.py will assert DataFrame.equals(expected_csv_df)
"""

import os
import sys
import glob
import argparse
import importlib.util
import traceback
import textwrap
import re
import json
from typing import Tuple

import pandas as pd
import pdfplumber

# --- Config ---
MAX_ATTEMPTS = 3
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CUSTOM_PARSERS_DIR = "custom_parsers"

# Utilities: file discovery
def find_target_files(target: str) -> Tuple[str, str]:
    folder = os.path.join("data", target)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing data folder: {folder}")
    
    pdfs = glob.glob(os.path.join(folder, "*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    
    sample_pdfs = [p for p in pdfs if "sample" in os.path.basename(p).lower()]
    pdf_path = sample_pdfs[0] if sample_pdfs else pdfs[0]
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    # Prefer result.csv if present
    preferred = [c for c in csvs if os.path.basename(c).lower().startswith("result")]
    csv_path = preferred[0] if preferred else csvs[0]
    return pdf_path, csv_path

# Diagnostics: sample text and word coordinates
def sample_pdf_text(pdf_path: str, max_pages: int = 2) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages[:max_pages]:
            texts.append(p.extract_text() or "")
    return "\n\n".join(texts)[:4000]

def sample_pdf_words_diag(pdf_path: str, pages: int = 2, max_lines: int = 40) -> str:
    """
    Returns a JSON-like string with page width/height and first N text lines
    each including words with x0,x1,top,bottom,text. Useful for LLM diagnostics.
    """
    pages_diag = []
    with pdfplumber.open(pdf_path) as pdf:
        for p_i, page in enumerate(pdf.pages[:pages]):
            try:
                words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
            except Exception:
                # fallback to extract_text split
                txt = page.extract_text() or ""
                pages_diag.append({
                    "page_index": p_i,
                    "width": float(page.width),
                    "height": float(page.height),
                    "text_sample": txt[:1000]
                })
                continue
            lines = {}
            for w in words:
                top = float(w.get("top", w.get("y0", 0)))
                left = float(w.get("x0", w.get("x", 0)))
                key = int(round(top/3) * 3)
                lines.setdefault(key, []).append({
                    "text": w.get("text"),
                    "x0": float(w.get("x0", w.get("x", 0))),
                    "x1": float(w.get("x1", w.get("x1", w.get("x0", 0) + 1))),
                    "top": top,
                    "bottom": float(w.get("bottom", w.get("y1", top + 5)))
                })
            sorted_lines = sorted(lines.items(), key=lambda kv: kv[0])[:max_lines]
            page_lines = []
            for key, wlist in sorted_lines:
                line_text = " ".join([ww["text"] for ww in sorted(wlist, key=lambda W: W["x0"])])
                page_lines.append({
                    "line_key": int(key),
                    "text": line_text,
                    "words": sorted(wlist, key=lambda W: W["x0"])
                })
            pages_diag.append({
                "page_index": p_i,
                "width": float(page.width),
                "height": float(page.height),
                "lines": page_lines
            })
    return json.dumps(pages_diag, indent=2)[:15000]

# Build the LLM prompt 
def build_generation_prompt(target: str, sample_text: str, csv_path: str, diag_words: str = None) -> str:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    sample_rows = df.head(8).to_csv(index=False)
    prompt = f"""
You are an expert Python engineer. Produce a single, valid Python module (one .py file) that implements:

    def parse(pdf_path: str) -> pandas.DataFrame

Requirements (STRICT):
- The returned DataFrame must have columns exactly and in order: {cols}
- The returned DataFrame must match the CSV at path: {csv_path} exactly (the agent will compare using pandas.DataFrame.equals)
- Use only: import pandas as pd, import pdfplumber, import re and Python stdlib
- DO NOT use 'vertical_lines' in table_settings or call extract_tables() relying on vertical_lines.
- Use pdfplumber.extract_words(...) to detect header & column x0 positions, group by 'top' (rounded) to form lines, then assign words to columns by x bounds.
- pdfplumber keys for coords are 'top'/'bottom' (some older code uses 'y0'/'y1' — agent will auto-patch those if present)
- The parser must:
    1) Find the header line containing 'Date' and 'Description' and 'Balance' (and 'Debit'/'Credit' words).
    2) Record x0 positions for each header (left boundaries), append page.width as rightmost boundary.
    3) Crop below header (use crop((0, header_bottom, page.width, page.height))) and extract_words() from the cropped page.
    4) Group words into rows by rounded 'top'.
    5) For each row, place each word into the correct column by checking x0 against header boundaries.
    6) Join adjacent words in same column with single spaces.
    7) Return a pandas.DataFrame with exact column names and types matching the CSV (strings for Date/Description, numeric for amounts if CSV shows numeric).
- The module must be valid Python and return the DataFrame. Output only one triple-backtick fenced Python code block (```python ... ```).

Short PDF sample (for structure):
{sample_text}

Example expected CSV rows (first 8):
{sample_rows}
"""
    if diag_words:
        prompt += "\n\nPDF diagnostics (word coords):\n" + diag_words + "\n\n"
    return textwrap.dedent(prompt)

# LLM clients 
def call_gemini(prompt: str, model: str = None) -> str:
    model = model or GEMINI_MODEL
    try:
        from google import genai
        client_cls = getattr(genai, "Client", None)
        if client_cls is None:
            raise RuntimeError("google-genai client has unexpected shape")
        client = client_cls()
        resp = client.models.generate_content(model=model, contents=prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}")

def call_groq(prompt: str, model: str = None) -> str:
    model = model or GROQ_MODEL
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        messages = [
            {"role": "system", "content": "You are a helpful Python engineer that returns a single Python module."},
            {"role": "user", "content": prompt},
        ]
        resp = client.chat.completions.create(messages=messages, model=model)
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq call failed: {e}")

def call_llm(prompt: str, provider: str = DEFAULT_PROVIDER) -> str:
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "gemini":
        return call_gemini(prompt)
    elif provider == "groq":
        return call_groq(prompt)
    else:
        raise RuntimeError(f"Unsupported provider {provider}. Set LLM_PROVIDER=gemini|groq or pass --provider.")

#  Code extraction & auto-patching 
def extract_code_from_response(resp_text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", resp_text, re.S | re.I)
    if m:
        code = m.group(1).strip()
    else:
        m2 = re.search(r"(def\s+parse\s*\(.*)", resp_text, re.S)
        code = resp_text[m2.start():].strip() if m2 else resp_text.strip()

    code = re.sub(r"^\s*```.*\n", "", code)
    code = re.sub(r"\n```\s*$", "", code).strip()

    code = re.sub(r"['\"]vertical_lines['\"]\s*:\s*[^,}\n]+,?", "", code)
    code = code.replace("['y0']", "['top']").replace("['y1']", "['bottom']")
    code = code.replace(".get('y0')", ".get('top')").replace(".get('y1')", ".get('bottom')")
    code = code.replace("pd.NA_dtype", "pd.NA")
    return code

# Write parser file 
def write_parser_file(code: str, target: str) -> str:
    os.makedirs(CUSTOM_PARSERS_DIR, exist_ok=True)
    path = os.path.join(CUSTOM_PARSERS_DIR, f"{target}_parser.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by agent.py\n")
        f.write(code + "\n")
    return path

# dynamic import 
def import_parser(module_path: str):
    spec = importlib.util.spec_from_file_location("parser_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# strict comparison
def compare_dfs_strict(df_out: pd.DataFrame, expected: pd.DataFrame) -> Tuple[bool, str]:
    try:
        ok = df_out.reset_index(drop=True).equals(expected.reset_index(drop=True))
        if ok:
            return True, "DataFrames match exactly (equals=True)"
        import pandas.testing as pdt
        try:
            pdt.assert_frame_equal(
                df_out.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False
            )
        except AssertionError as e:
            return False, str(e)
        return False, "DataFrames differ in subtle way"
    except Exception as e:
        return False, f"Comparison error: {e}"

# main agent loop 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Target folder name inside data/ (e.g. icici)")
    ap.add_argument("--provider", required=False, help="LLM provider: gemini or groq (overrides env var)")
    args = ap.parse_args()
    provider = (args.provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()

    try:
        pdf_path, csv_path = find_target_files(args.target)
    except Exception as e:
        print("Error locating data files:", e)
        sys.exit(2)

    print("Using provider:", provider)
    print("PDF:", pdf_path)
    print("CSV:", csv_path)

    sample_text = sample_pdf_text(pdf_path, max_pages=2)
    diag_words = sample_pdf_words_diag(pdf_path, pages=2, max_lines=40)
    expected_df = pd.read_csv(csv_path)

    attempt = 0
    last_code = None
    last_error = None

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"\n=== Attempt {attempt}/{MAX_ATTEMPTS} ===")
        if last_code is None:
            prompt = build_generation_prompt(args.target, sample_text, csv_path, diag_words=diag_words)
        else:
            hint = ""
            if isinstance(last_error, str) and "(0," in last_error:
                hint = "Previous parser returned empty DataFrame — header detection or row extraction likely failed. Use the diagnostics to locate the header line and x0 boundaries."
            else:
                hint = f"Previous failure details:\n{last_error}"

            prompt = (
                f"The previous parser code failed tests.\n{hint}\n\n"
                f"Here is the previous parser code (please provide a corrected single-file Python module):\n```python\n{last_code}\n```\n"
                "Do NOT use vertical_lines. Use extract_words(), group by 'top', detect header by 'Date'/'Description'/'Balance' and record x0 positions, crop, then split rows by x-boundaries. Return a DataFrame that matches the CSV exactly."
            )
            prompt += "\n\nPDF diagnostics (word coords):\n" + diag_words

        print("Calling LLM to generate parser...")
        try:
            resp = call_llm(prompt, provider=provider)
        except Exception as e:
            print("LLM call failed:", e)
            traceback.print_exc()
            sys.exit(3)

        code = extract_code_from_response(resp)
        last_code = code
        parser_file = write_parser_file(code, args.target)
        print("Wrote parser to:", parser_file)

        try:
            module = import_parser(parser_file)
            if not hasattr(module, "parse"):
                last_error = "Generated module does not define function parse(pdf_path)"
                print("ERROR:", last_error)
                continue
            df_out = module.parse(pdf_path)
            if not isinstance(df_out, pd.DataFrame):
                last_error = f"parse() did not return a pandas.DataFrame (got {type(df_out)})"
                print("ERROR:", last_error)
                continue

            ok, info = compare_dfs_strict(df_out, expected_df)
            if ok:
                print("\nSUCCESS — parser produced DataFrame matching CSV exactly.")
                print(info)
                try:
                    import subprocess
                    subprocess.run(["pytest", "-q"], check=False)
                except Exception:
                    pass
                return
            else:
                last_error = info
                print("Test failed (DataFrame mismatch):")
                print(info)
                try:
                    last_error += "\n\nSample output rows (first 10):\n" + df_out.head(10).to_csv(index=False)
                except Exception:
                    pass

        except Exception as e:
            tb = traceback.format_exc()
            last_error = f"Exception while importing/running parser:\n{tb}"
            print("Exception trace:\n", tb)

    print("\nAgent exhausted attempts. Last failure:\n", last_error)
    sys.exit(2)

if __name__ == "__main__":
    main()
