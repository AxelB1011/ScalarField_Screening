# utilities.py - Clean Helper Functions

import re
import os
from typing import Optional, Tuple, Dict
from datetime import datetime, date
from dateutil.parser import parse as dtparse
from sec_cik_mapper import StockMapper
from functools import lru_cache

# Initialize mapper
_mapper = StockMapper()

@lru_cache(maxsize=4096)
def cik_to_ticker(cik: str) -> Optional[str]:
    """Convert CIK to ticker symbol"""
    try:
        ticker_set = _mapper.cik_to_tickers.get(cik.zfill(10))
        if ticker_set:
            return list(ticker_set)[0]
    except Exception:
        pass
    return None

@lru_cache(maxsize=4096)
def ticker_to_cik(ticker: str) -> Optional[str]:
    """Convert ticker to CIK"""
    try:
        return _mapper.ticker_to_cik.get(ticker)
    except Exception:
        pass
    return None

def _extract_xbrl_fiscal_info(filing) -> Tuple[Optional[int], Optional[int]]:
    """Extract fiscal year and quarter from XBRL"""
    try:
        # Method 1: Use XBRL facts
        doc_period_df = filing.xbrl().facts.query().by_concept("DocumentPeriodEndDate").to_dataframe()
        if not doc_period_df.empty:
            end_date_str = doc_period_df["value"].iloc[0]
            end_date = dtparse(end_date_str).date()
            fiscal_year = end_date.year
            
            # Try to get fiscal period focus
            try:
                period_focus_df = filing.xbrl().facts.query().by_concept("DocumentFiscalPeriodFocus").to_dataframe()
                if not period_focus_df.empty:
                    period_focus = period_focus_df["value"].iloc[0].strip().upper()
                    if period_focus == "FY":
                        fiscal_quarter = None
                    elif period_focus.startswith("Q"):
                        fiscal_quarter = int(period_focus[1])
                    else:
                        fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
                else:
                    fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
            except Exception:
                fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
            
            return fiscal_year, fiscal_quarter
    except Exception:
        pass
    
    try:
        # Method 2: Use search_facts as fallback
        doc_period_results = filing.xbrl().facts.search_facts("DocumentPeriodEndDate")
        if doc_period_results and len(doc_period_results) > 0:
            end_date_str = doc_period_results[0].value
            end_date = dtparse(end_date_str).date()
            fiscal_year = end_date.year
            
            # Try DocumentFiscalPeriodFocus with search_facts
            try:
                period_focus_results = filing.xbrl().facts.search_facts("DocumentFiscalPeriodFocus")
                if period_focus_results and len(period_focus_results) > 0:
                    period_focus = period_focus_results[0].value.strip().upper()
                    if period_focus == "FY":
                        fiscal_quarter = None
                    elif period_focus.startswith("Q"):
                        fiscal_quarter = int(period_focus[1])
                    else:
                        fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
                else:
                    fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
            except Exception:
                fiscal_quarter = ((end_date.month - 1) // 3 + 1) if filing.form.upper() == "10-Q" else None
            
            return fiscal_year, fiscal_quarter
    except Exception:
        pass
    
    return None, None

def _extract_fiscal_info(text: str, filing=None) -> Tuple[Optional[int], Optional[int]]:
    """Extract fiscal year and quarter information"""
    # Try XBRL first if available
    if filing is not None:
        fiscal_year, fiscal_quarter = _extract_xbrl_fiscal_info(filing)
        if fiscal_year is not None:
            return fiscal_year, fiscal_quarter
    
    # Regex fallback on text
    header = " ".join(text[:8000].split())  # normalize whitespace
    
    # Try fiscal year pattern
    m = re.search(r'for\s+(?:the\s+)?fiscal\s+year\s+ended\s+([\w\s,]{5,40})', header, re.I)
    if m:
        try:
            fy = dtparse(m.group(1)).year
            return fy, None
        except Exception:
            pass
    
    # Try quarter pattern
    m = re.search(r'for\s+(?:the\s+)?quarter\s+ended\s+([\w\s,]{5,40})', header, re.I)
    if m:
        try:
            end_dt = dtparse(m.group(1))
            fy = end_dt.year
            fq = (end_dt.month - 1) // 3 + 1
            return fy, fq
        except Exception:
            pass
    
    return None, None

def _build_document_url(filing) -> str:
    """Construct direct document URL"""
    try:
        cik = str(filing.cik)
        accession = str(filing.accession_no).replace('-', '')
        primary_doc = getattr(filing, 'primary_document', '')
        is_xbrl = getattr(filing, 'is_inline_xbrl', False)
        
        if not primary_doc:
            return getattr(filing, 'url', '')
        
        base_path = f"Archives/edgar/data/{cik}/{accession}/{primary_doc}"
        prefix = "ix?doc=/" if is_xbrl else ""
        
        return f"https://www.sec.gov/{prefix}{base_path}"
    except Exception:
        return getattr(filing, 'url', '')

def build_base_metadata(filing, main_text: str) -> Dict:
    """Build complete metadata dict for chunker/DB insertion"""
    fiscal_year, fiscal_quarter = _extract_fiscal_info(main_text, filing)
    
    return {
        "cik": str(filing.cik),
        "ticker": cik_to_ticker(str(filing.cik)) or '',
        "form_type": filing.form,
        "filing_date": str(filing.filing_date),
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "filing_url": getattr(filing, 'url', ''),
        "document_url": _build_document_url(filing),
    }