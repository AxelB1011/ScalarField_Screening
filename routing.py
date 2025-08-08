# routing.py - Clean Routing System

import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sec_cik_mapper import StockMapper

# Initialize mapper
_mapper = StockMapper()

class FormType(Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"  
    FORM_8K = "8-K"
    FORM_DEF14A = "DEF14A"

@dataclass(frozen=True)
class SectionIdentifier:
    form_type: FormType
    section_type: str  # ITEM, PART, etc.
    section_number: str  # 1A, I, 2.02, etc.
    
    def __str__(self) -> str:
        return f"{self.form_type.value} {self.section_type} {self.section_number}".strip()

@dataclass
class ConceptMapping:
    """Maps concepts to sections across forms"""
    concept: str
    sections: List[SectionIdentifier]
    confidence: float = 1.0
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

@dataclass
class RoutingResult:
    """Result of routing with recommendations"""
    primary_targets: List[SectionIdentifier]
    confidence_scores: Dict[SectionIdentifier, float]
    reasoning: str
    recommended_tickers: List[str]
    recommended_forms: List[FormType]

class TemporalScope(Enum):
    SPECIFIC_YEAR = "specific_year"
    RECENT = "recent"
    HISTORICAL = "historical"
    ANNUAL = "annual"
    QUARTERLY = "quarterly"

@dataclass
class TemporalContext:
    scope: TemporalScope
    specific_year: Optional[int] = None
    year_range: Optional[Tuple[int, int]] = None

@dataclass
class QueryContext:
    tickers: List[str]
    temporal_context: TemporalContext
    original_query: str
    is_comparison: bool = False

class ConceptRepository:
    """Repository of financial concepts mapped to SEC sections"""
    
    def __init__(self):
        self._concepts: Dict[str, ConceptMapping] = {}
        self._load_concepts()
    
    def _load_concepts(self):
        """Load financial concept mappings"""
        mappings = [
            # Financial statements
            ConceptMapping(
                "financial_statements",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "8"),
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "1"),
                    SectionIdentifier(FormType.FORM_8K, "ITEM", "2.02"),   # 8-K financial exhibits
                ],
                confidence=0.95,
                aliases=[
                    "income_statement", "consolidated_statements", "balance_sheet",
                    "cash_flow", "financial_position", "revenue", "earnings",
                    "profit", "assets", "liabilities", "equity"
                ]
            ),

            # Research & Development
            ConceptMapping(
                "research_development",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "1"),
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "7"),
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "1"),
                ],
                confidence=0.90,
                aliases=[
                    "r&d", "research", "development", "innovation",
                    "rd_spending", "research_expenses", "development_costs"
                ]
            ),

            # Risk factors
            ConceptMapping(
                "risk_factors",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "1A"),
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "3"),     # 10-Q Risk Factors
                ],
                confidence=0.95,
                aliases=["risk", "risks", "risk_factors", "threats", "challenges"]
            ),

            # Business overview / company description
            ConceptMapping(
                "business_overview",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "1"),
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "1"),
                ],
                confidence=0.90,
                aliases=["business", "overview", "operations", "description", "company profile"]
            ),

            # MD&A
            ConceptMapping(
                "mda",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "7"),
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "2"),
                ],
                confidence=0.95,
                aliases=["md&a", "management", "discussion", "analysis"]
            ),

            # Insider Transactions (Forms 3,4,5)
            ConceptMapping(
                "insider_transactions",
                [
                    SectionIdentifier(FormType.FORM_DEF14A, "ITEM", "5"),  # DEF 14A Proxy: Section on Insider Ownership
                    SectionIdentifier(FormType.FORM_8K, "ITEM", "5"),     # 8-K Item 5 often covers specific disclosures
                ],
                confidence=0.85,
                aliases=["insider trading", "form 4", "form 5", "form 3", "beneficial ownership"]
            ),

            # Material Agreements
            ConceptMapping(
                "material_agreements",
                [
                    SectionIdentifier(FormType.FORM_8K, "ITEM", "1.01"),  # Entry into a material agreement
                    SectionIdentifier(FormType.FORM_DEF14A, "ITEM", "2"), # DEF 14A: Proposal description often includes agreements
                ],
                confidence=0.90,
                aliases=["material agreement", "contract", "merger agreement", "transaction agreement"]
            ),

            # Corporate Governance (DEF 14A)
            ConceptMapping(
                "corporate_governance",
                [
                    SectionIdentifier(FormType.FORM_DEF14A, "ITEM", "7"), # DEF 14A: Corporate governance disclosures
                ],
                confidence=0.90,
                aliases=["governance", "board", "committee", "director independence"]
            ),

            # Dividends and Distributions
            ConceptMapping(
                "dividends",
                [
                    SectionIdentifier(FormType.FORM_10K, "ITEM", "5"),    # Market for registrant’s common equity; includes dividends
                    SectionIdentifier(FormType.FORM_10Q, "ITEM", "6"),    # Exhibits that may include dividend notices
                ],
                confidence=0.85,
                aliases=["dividend", "distribution", "shareholder payment", "dividends paid"]
            ),
        ]
        
        for mapping in mappings:
            self._concepts[mapping.concept] = mapping
    
    def get_concept(self, concept_name: str) -> Optional[ConceptMapping]:
        return self._concepts.get(concept_name)
    
    def get_all_concepts(self) -> Dict[str, ConceptMapping]:
        return self._concepts.copy()

class TickerExtractor:
    """Extract ticker symbols from queries"""
    
    # Company name to ticker mapping
    COMPANY_NAME_MAP = {
        r'\b(?:apple|apple\s+inc\.?)\b': 'AAPL',
        r'\b(?:microsoft|microsoft\s+corp\.?)\b': 'MSFT',
        r'\b(?:google|alphabet|alphabet\s+inc\.?)\b': 'GOOGL',
        r'\b(?:amazon|amazon\.com)\b': 'AMZN',
        r'\b(?:tesla|tesla\s+inc\.?)\b': 'TSLA',
        r'\b(?:meta|facebook|meta\s+platforms)\b': 'META',
        r'\b(?:nvidia|nvidia\s+corp\.?)\b': 'NVDA',
    }
    
    # Valid ticker symbols
    VALID_TICKERS = set(_mapper.ticker_to_cik.keys())
    
    # Words to exclude
    EXCLUDED_WORDS = {
        'WHAT', 'ARE', 'THE', 'AND', 'FOR', 'HAS', 'HOW', 'OVER', 'TIME',
        'WILL', 'CAN', 'MAY', 'SHOULD', 'RISK', 'FROM', 'WITH'
    }
    
    def extract(self, query: str) -> Tuple[List[str], bool]:
        """Extract tickers and determine if comparison query"""
        found_tickers = set()
        query_lower = query.lower()
        
        # Extract by company name patterns
        for pattern, ticker in self.COMPANY_NAME_MAP.items():
            if re.search(pattern, query_lower):
                found_tickers.add(ticker)
        
        # Extract explicit ticker mentions
        ticker_patterns = [
            r'\b([A-Z]{2,5})[\'\s]s\b',  # AAPL's
            r'\(([A-Z]{2,5})\)',  # (AAPL)
            r'\b([A-Z]{2,5})\s+stock\b',  # AAPL stock
        ]
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if match in self.VALID_TICKERS and match not in self.EXCLUDED_WORDS:
                    found_tickers.add(match)
        
        # Default fallback if no tickers found
        if not found_tickers:
            found_tickers.add('AAPL')  # Default to Apple
        
        # Check for comparison indicators
        comparison_indicators = [
            'compare', 'comparison', 'vs', 'versus', 'against',
            'both', 'between', 'and', 'difference'
        ]
        is_comparison = any(indicator in query_lower for indicator in comparison_indicators)
        is_comparison = is_comparison or len(found_tickers) > 1
        
        return list(found_tickers), is_comparison

class TemporalExtractor:
    """Extract temporal context from queries"""
    
    def extract(self, query: str) -> TemporalContext:
        """Extract temporal context"""
        query_lower = query.lower()
        current_year = datetime.now().year
        
        # Specific year patterns
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            return TemporalContext(
                scope=TemporalScope.SPECIFIC_YEAR,
                specific_year=year
            )
        
        # Historical trend indicators
        trend_indicators = [
            'over time', 'trend', 'historical', 'changed', 'evolution',
            'how has', 'year over year', 'historically'
        ]
        if any(indicator in query_lower for indicator in trend_indicators):
            return TemporalContext(
                scope=TemporalScope.HISTORICAL,
                year_range=(current_year - 5, current_year)
            )
        
        # Recent indicators
        recent_indicators = ['recent', 'latest', 'current', 'new', 'last']
        if any(indicator in query_lower for indicator in recent_indicators):
            return TemporalContext(
                scope=TemporalScope.RECENT,
                year_range=(current_year - 1, current_year)
            )
        
        # Annual indicators
        if any(word in query_lower for word in ['annual', 'yearly', '10-k']):
            return TemporalContext(scope=TemporalScope.ANNUAL)
        
        # Quarterly indicators
        if any(word in query_lower for word in ['quarterly', 'quarter', '10-q']):
            return TemporalContext(scope=TemporalScope.QUARTERLY)
        
        # Default to recent
        return TemporalContext(
            scope=TemporalScope.RECENT,
            year_range=(current_year - 1, current_year)
        )

class Router:
    """Main routing class"""
    
    def __init__(self):
        self.repository = ConceptRepository()
        self.ticker_extractor = TickerExtractor()
        self.temporal_extractor = TemporalExtractor()
    
    def route_query(self, query: str) -> RoutingResult:
        """Route query to relevant sections"""
        # Extract context
        tickers, is_comparison = self.ticker_extractor.extract(query)
        temporal_context = self.temporal_extractor.extract(query)
        
        query_context = QueryContext(
            tickers=tickers,
            temporal_context=temporal_context,
            original_query=query,
            is_comparison=is_comparison
        )
        
        # Find matching concepts
        concept_matches = self._find_concept_matches(query.lower())
        
        # Score sections
        section_scores = self._calculate_section_scores(
            concept_matches, temporal_context, query.lower()
        )
        
        # Sort by score
        sorted_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
        primary_targets = [section for section, score in sorted_sections]
        
        # Determine recommended forms
        recommended_forms = self._get_recommended_forms(temporal_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(concept_matches, temporal_context, primary_targets)
        
        return RoutingResult(
            primary_targets=primary_targets,
            confidence_scores=dict(sorted_sections),
            reasoning=reasoning,
            recommended_tickers=tickers,
            recommended_forms=recommended_forms
        )
    
    def _find_concept_matches(self, query_lower: str) -> List[Tuple[str, ConceptMapping, float]]:
        """Find concepts that match the query"""
        matches = []
        
        for concept_name, mapping in self.repository.get_all_concepts().items():
            confidence = 0.0
            
            # Direct concept name match
            if concept_name.replace("_", " ") in query_lower:
                confidence = 1.0
            # Alias matching
            elif any(alias.replace("_", " ") in query_lower for alias in mapping.aliases):
                confidence = 0.9
            
            if confidence > 0:
                matches.append((concept_name, mapping, confidence))
        
        return matches
    
    def _calculate_section_scores(
        self, concept_matches: List[Tuple[str, ConceptMapping, float]], 
        temporal_context: TemporalContext, 
        query: str
    ) -> Dict[SectionIdentifier, float]:
        """Calculate scores for each section"""
        section_scores = {}
        
        for concept_name, mapping, match_confidence in concept_matches:
            base_score = match_confidence * mapping.confidence
            
            for section in mapping.sections:
                # Apply temporal boost
                temporal_boost = self._get_temporal_boost(section.form_type, temporal_context)
                
                # Apply query-specific boosts
                query_boost = self._get_query_boost(section, query)
                
                final_score = base_score * temporal_boost * query_boost
                section_scores[section] = max(section_scores.get(section, 0.0), final_score)
        
        return section_scores
    
    def _get_temporal_boost(self, form_type: FormType, temporal_context: TemporalContext) -> float:
        """Apply temporal alignment boost"""
        scope = temporal_context.scope
        
        if scope in [TemporalScope.SPECIFIC_YEAR, TemporalScope.ANNUAL]:
            if form_type in [FormType.FORM_10K, FormType.FORM_DEF14A]:
                return 1.2
        elif scope == TemporalScope.QUARTERLY:
            if form_type == FormType.FORM_10Q:
                return 1.2
        elif scope == TemporalScope.RECENT:
            if form_type == FormType.FORM_8K:
                return 1.2
        
        return 1.0
    
    def _get_query_boost(self, section: SectionIdentifier, query: str) -> float:
        """Apply query-specific boosts"""
        boost = 1.0
        
        # Financial data boosts
        if any(word in query for word in ["revenue", "income", "profit", "earnings"]):
            if section.section_number == "8":
                boost *= 1.3
        
        # R&D boosts
        if any(word in query for word in ["r&d", "research", "development"]):
            if section.section_number in ["7", "8"]:
                boost *= 1.3
        
        return boost
    
    def _get_recommended_forms(self, temporal_context: TemporalContext) -> List[FormType]:
        """Get recommended form types"""
        scope = temporal_context.scope
        
        if scope in [TemporalScope.SPECIFIC_YEAR, TemporalScope.ANNUAL, TemporalScope.HISTORICAL]:
            return [FormType.FORM_10K, FormType.FORM_10Q]
        elif scope == TemporalScope.QUARTERLY:
            return [FormType.FORM_10Q]
        elif scope == TemporalScope.RECENT:
            return [FormType.FORM_8K, FormType.FORM_10Q]
        
        return [FormType.FORM_10K, FormType.FORM_10Q]
    
    def _generate_reasoning(
        self, concept_matches: List[Tuple[str, ConceptMapping, float]], 
        temporal_context: TemporalContext, 
        primary_targets: List[SectionIdentifier]
    ) -> str:
        """Generate human-readable reasoning"""
        reasoning_parts = []
        
        if concept_matches:
            top_concept = concept_matches[0][0].replace("_", " ").title()
            reasoning_parts.append(f"Identified primary concept: {top_concept}")
        
        reasoning_parts.append(f"Temporal strategy: {temporal_context.scope.value}")
        
        if primary_targets:
            sections = [str(target) for target in primary_targets[:3]]
            reasoning_parts.append(f"Best matches: {', '.join(sections)}")
        
        return ". ".join(reasoning_parts)