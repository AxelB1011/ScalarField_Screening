# chunking.py - Clean SEC Document Chunking System

import re
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict, Optional
from dateutil.parser import parse as dtparse
from datetime import date, datetime
from utilities import *
from edgar import *
from dotenv import load_dotenv

load_dotenv()
set_identity(os.getenv("EDGAR_IDENTITY"))

@dataclass
class FormChunk:
    """Clean FormChunk with all essential metadata"""
    form_type: str
    section_type: str
    section_number: str
    section_title: str
    content: str
    start_pos: int
    end_pos: int
    
    # Company metadata
    cik: str = ""
    ticker: str = ""
    filing_date: str = ""
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    
    # Document metadata
    chunk_id: str = ""
    content_type: str = ""
    char_count: int = 0
    filing_url: str = ""
    document_url: str = ""
    
    # Attachment metadata
    parent_form_type: Optional[str] = None
    parent_filing_date: Optional[str] = None
    attachment_number: str = ""
    attachment_description: str = ""
    is_attachment: bool = False
    attachment_type: str = ""
    
    # Vector storage
    embedding: Optional[bytes] = None
    
    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.chunk_id:
            self._generate_chunk_id()
    
    def _generate_chunk_id(self):
        """Generate unique chunk ID"""
        identifier = self.cik if self.cik else self.ticker
        base = f"{identifier}_{self.form_type}_{self.fiscal_year or 'UNK'}"
        
        if self.is_attachment:
            attachment_id = self.attachment_number or hash(self.document_url) % 10000
            self.chunk_id = f"{base}_ATT{attachment_id}_{hash(self.content) % 10000:04d}"
        else:
            self.chunk_id = f"{base}_{self.section_number}_{hash(self.content) % 10000:04d}"

def _pre(text: str) -> str:
    """Preprocessing"""
    return text.replace("\r\n", "\n")

# Form parsing patterns
_pat_10k = re.compile(
    r'(?i)(?:^|\n)\s*'
    r'(?:(PART\s+[IVX]+)\s*[\.\-]?\s*([^\n\r]*?)(?:\n|$))'
    r'|(?:^|\n)\s*(?:ITEM\s+(\d+[A-Z]?)\s*[\.\-]?\s*([^\n\r]+))',
    re.MULTILINE
)

def _parse_10k(m: re.Match) -> Tuple[str, str, str]:
    if m.group(1):
        return "PART", m.group(1), (m.group(2) or "").strip()
    return "ITEM", m.group(3), (m.group(4) or "").strip()

_pat_10q = re.compile(
    r'(?i)(?:^|\n)\s*'
    r'(?:(PART\s+[IVX]+)\s*[\.\-]?\s*([^\n\r]*?)(?:\n|$))'
    r'|(?:^|\n)\s*(?:ITEM\s+(\d+(?:\-[A-Z])?)\s*[\.\-]?\s*([^\n\r]+))',
    re.MULTILINE
)

def _parse_10q(m):
    if m.group(1):
        return "PART", m.group(1), (m.group(2) or "").strip()
    return "ITEM", m.group(3), (m.group(4) or "").strip()

_pat_8k = re.compile(
    r'(?i)(?:^|\n)\s*'
    r'(?:ITEM\s+(\d+(?:\.\d+)?[A-Z]?)\s*[\.\-]?\s*([^\n\r]+))'
    r'|(?:^|\n)\s*(SIGNATURE[S]?)\s*$',
    re.MULTILINE
)

def _parse_8k(m):
    if m.group(1):
        return "ITEM", m.group(1), (m.group(2) or "").strip()
    return "SIGNATURE", "", m.group(3).strip()

# Registry mapping form types to patterns
_REGISTRY: Dict[str, Tuple[re.Pattern, Callable[[re.Match], Tuple[str, str, str]]]] = {
    "10-K": (_pat_10k, _parse_10k),
    "10-Q": (_pat_10q, _parse_10q),
    "8-K": (_pat_8k, _parse_8k),
}

def chunk_form(text: str, form_type: str, **metadata) -> List[FormChunk]:
    """
    Enhanced chunker with metadata support
    
    Args:
        text: Document text
        form_type: SEC form type
        **metadata: Additional metadata (cik, ticker, filing_date, etc.)
    
    Returns:
        List of FormChunk objects
    """
    form_type = form_type.upper().replace(" ", "")
    
    if form_type not in _REGISTRY:
        # For unsupported forms, return single chunk
        return [FormChunk(
            form_type=form_type,
            section_type="DOCUMENT",
            section_number="",
            section_title="Entire Document",
            content=text,
            start_pos=0,
            end_pos=len(text),
            **metadata
        )]
    
    pattern, parser = _REGISTRY[form_type]
    text = _pre(text)
    matches = list(pattern.finditer(text))
    chunks: List[FormChunk] = []
    
    # Extract metadata
    cik = metadata.get('cik', '')
    ticker = metadata.get('ticker', '')
    filing_date = metadata.get('filing_date', '')
    fiscal_year = metadata.get('fiscal_year')
    fiscal_quarter = metadata.get('fiscal_quarter')
    filing_url = metadata.get('filing_url', '')
    document_url = metadata.get('document_url', '')
    is_attachment = metadata.get('is_attachment', False)
    attachment_type = metadata.get('attachment_type', '')
    parent_form_type = metadata.get('parent_form_type')
    parent_filing_date = metadata.get('parent_filing_date')
    
    # Add header chunk if content exists before first match
    if matches and matches[0].start() > 0:
        chunks.append(FormChunk(
            form_type=form_type,
            section_type="HEADER",
            section_number="",
            section_title="Document Header",
            content=text[:matches[0].start()],
            start_pos=0,
            end_pos=matches[0].start(),
            cik=cik,
            ticker=ticker,
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            filing_url=filing_url,
            document_url=document_url,
            is_attachment=is_attachment,
            attachment_type=attachment_type,
            parent_form_type=parent_form_type,
            parent_filing_date=parent_filing_date,
        ))
    
    # Process matched sections
    for i, m in enumerate(matches):
        section_type, section_number, section_title = parser(m)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        content_type = _infer_content_type(section_type, section_number, section_title)
        
        chunk = FormChunk(
            form_type=form_type,
            section_type=section_type,
            section_number=section_number,
            section_title=section_title,
            content=text[start:end],
            start_pos=start,
            end_pos=end,
            cik=cik,
            ticker=ticker,
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            content_type=content_type,
            filing_url=filing_url,
            document_url=document_url,
            is_attachment=is_attachment,
            attachment_type=attachment_type,
            parent_form_type=parent_form_type,
            parent_filing_date=parent_filing_date,
        )
        
        if is_attachment:
            chunk.attachment_number = section_number
            chunk.attachment_description = section_title
            
        chunks.append(chunk)
    
    # If no matches found, create single document chunk
    if not chunks:
        chunks.append(FormChunk(
            form_type=form_type,
            section_type="DOCUMENT",
            section_number="",
            section_title="Entire Document",
            content=text,
            start_pos=0,
            end_pos=len(text),
            cik=cik,
            ticker=ticker,
            filing_date=filing_date,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            filing_url=filing_url,
            document_url=document_url,
            is_attachment=is_attachment,
            attachment_type=attachment_type,
            parent_form_type=parent_form_type,
            parent_filing_date=parent_filing_date,
        ))
    
    return chunks

def _infer_content_type(section_type: str, section_number: str, section_title: str) -> str:
    """Infer content type based on section info"""
    title_lower = section_title.lower()
    
    # Financial statement detection
    financial_indicators = [
        "consolidated statements", "statement of operations", "income statement",
        "balance sheet", "cash flow", "financial position", "shareholders equity",
        "consolidated balance", "consolidated income", "statements of earnings"
    ]
    if any(phrase in title_lower for phrase in financial_indicators):
        return "financials"
    
    # MD&A detection
    if "md&a" in title_lower or "management's discussion" in title_lower:
        return "mda"
    
    # Business overview
    if any(word in title_lower for word in ["business", "overview", "operations"]):
        return "business_overview"
    
    # Section-specific mappings
    if section_type == "ITEM":
        if section_number == "1A" or "risk" in title_lower:
            return "risk_factors"
        elif section_number == "1" or "business" in title_lower:
            return "business_overview"
        elif section_number == "7" or "md&a" in title_lower:
            return "mda"
        elif section_number == "8" or any(phrase in title_lower for phrase in financial_indicators):
            return "financials"
        elif section_number in ["10", "11"] or "compensation" in title_lower:
            return "compensation"
    
    return "general"

def process_filing_attachments(filing: object, base_metadata: Dict) -> List[FormChunk]:
    """Process filing attachments and return chunks"""
    all_chunks = []
    
    if not hasattr(filing, 'attachments'):
        return all_chunks
    
    # parent_form_type = base_metadata.get('form_type')
    parent_form_type = base_metadata.pop('form_type', None)
    
    for attachment in filing.attachments:
        # Skip non-useful attachments
        if (attachment.purpose is None) and (attachment.extension != ".htm"):
            continue
        
        try:
            attachment_text = attachment.text()
            if not attachment_text or len(attachment_text.strip()) < 50:
                continue
            
            attachment_type = _determine_attachment_type(attachment)
            
            attachment_metadata = base_metadata.copy()
            attachment_metadata.update({
                'is_attachment': True,
                'attachment_type': attachment_type,
                'parent_form_type': parent_form_type,
                'parent_filing_date': base_metadata.get('filing_date'),
                'document_url': getattr(attachment, 'url', ''),
            })
            
            chunks = [FormChunk(
                form_type=attachment_type.upper(),
                section_type="DOCUMENT",
                section_number="",
                section_title=getattr(attachment, 'description', 'Attachment'),
                content=attachment_text,
                start_pos=0,
                end_pos=len(attachment_text),
                **attachment_metadata
            )]
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error processing attachment {attachment.document}: {e}")
            continue
    
    return all_chunks

def _determine_attachment_type(attachment) -> str:
    """Determine attachment type from document name and description"""
    doc_name = getattr(attachment, 'document', '').upper()
    description = getattr(attachment, 'description', '').upper()
    
    if doc_name.startswith('EX-'):
        if 'EX-99' in doc_name:
            return "press_release"
        elif 'EX-10' in doc_name:
            return "material_agreement"
        elif 'EX-21' in doc_name:
            return "subsidiaries"
        else:
            return "exhibit"
    
    if 'PRESS RELEASE' in description:
        return "press_release"
    elif 'AGREEMENT' in description or 'CONTRACT' in description:
        return "material_agreement"
    elif 'FINANCIAL STATEMENTS' in description:
        return "financials"
    elif 'EXHIBIT' in description:
        return "exhibit"
    
    return "attachment"