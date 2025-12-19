"""
Entity extraction from text using heuristic methods.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    import spacy  # type: ignore
    
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    start_char: int
    end_char: int
    confidence: float
    entity_type: str = "UNKNOWN"


class EntityExtractor:
    """Extracts entities from text using heuristic methods."""
    
    def __init__(self, config, use_spacy: bool = False):
        """
        Initialize entity extractor.
        
        Args:
            config: Configuration object
            use_spacy: Whether to use spaCy NER as fallback
        """
        self.config = config
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.use_spacy = False
                print("Warning: spaCy model not found. Using heuristic extraction only.")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with positions and confidence scores
        """
        entities = []
        
        # Heuristic extraction: capitalized phrases
        entities.extend(self._extract_capitalized_phrases(text))
        
        # Acronym detection
        entities.extend(self._extract_acronyms(text))
        
        # Optional: spaCy NER fallback
        if self.use_spacy and self.nlp:
            entities.extend(self._extract_with_spacy(text))
        
        # Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_capitalized_phrases(self, text: str) -> List[Entity]:
        """Extract capitalized phrases as potential entities."""
        entities = []
        
        # Pattern: Capitalized word(s) optionally followed by lowercase words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]{2,})?\b'
        
        for match in re.finditer(pattern, text):
            phrase = match.group()
            if self.config.entity_min_length <= len(phrase) <= self.config.entity_max_length:
                # Confidence based on length and capitalization pattern
                confidence = self._calculate_confidence(phrase)
                entities.append(
                    Entity(
                        text=phrase,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=confidence,
                        entity_type="CAPITALIZED_PHRASE",
                    )
                )
        
        return entities
    
    def _extract_acronyms(self, text: str) -> List[Entity]:
        """Extract acronyms (2-5 uppercase letters)."""
        entities = []
        
        pattern = r'\b[A-Z]{2,5}\b'
        
        for match in re.finditer(pattern, text):
            acronym = match.group()
            # Check if it's not a common word (e.g., "THE", "AND")
            if len(acronym) >= 2 and acronym not in {"THE", "AND", "FOR", "ARE", "BUT"}:
                confidence = self.config.acronym_expansion_confidence
                entities.append(
                    Entity(
                        text=acronym,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=confidence,
                        entity_type="ACRONYM",
                    )
                )
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if self.config.entity_min_length <= len(ent.text) <= self.config.entity_max_length:
                entities.append(
                    Entity(
                        text=ent.text,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=0.9,  # spaCy entities are generally high confidence
                        entity_type=ent.label_,
                    )
                )
        
        return entities
    
    def _calculate_confidence(self, phrase: str) -> float:
        """
        Calculate confidence score for an extracted entity.
        
        Args:
            phrase: Extracted phrase
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.7
        
        # Boost for longer phrases (likely proper nouns)
        if len(phrase.split()) >= 2:
            confidence += 0.1
        
        # Boost for all caps (likely acronyms or important entities)
        if phrase.isupper() and len(phrase) >= 2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove overlapping entities, keeping the one with highest confidence.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list
        """
        if not entities:
            return []
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        result = []
        
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in result:
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(entity)
        
        return result
    
    def _entities_overlap(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap in position."""
        return not (e1.end_char <= e2.start_char or e2.end_char <= e1.start_char)

