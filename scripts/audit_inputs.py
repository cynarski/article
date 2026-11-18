"""Audit precomputed rolling-fit inputs without recomputing the model."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lppls_research.quality import audit_input_files

def main():
    """Check provenance, calendar coverage and LPPLS acceptance rates."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices',default=str(ROOT/'btc_usd_2015_2024.csv'))
    parser.add_argument('--fits',default=str(ROOT/'lppls_rolling_results.csv'))
    parser.add_argument('--expected-fits',type=int,default=107240)
    parser.add_argument('--output',default=str(ROOT/'input_audit.json'))
    args=parser.parse_args()
    audit=audit_input_files(args.prices,args.fits,
               expected_fits=args.expected_fits,require_daily_prices=True)
    Path(args.output).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    main()
