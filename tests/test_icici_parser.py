import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from custom_parsers.icici_parser import parse

def test_icici_parser_matches_csv():
    pdf_path = "data/icici/icici sample.pdf"  # adjust if needed
    csv_path = "data/icici/result.csv"

    expected_df = pd.read_csv(csv_path)
    output_df = parse(pdf_path)

    assert output_df.equals(expected_df), "Parser output does not match the expected CSV"
