"""
Entity resolution and normalization.
"""
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EntityVariant:
    """Represents a variant of an entity."""
    text: str
    canonical_id: str
    confidence: float


class EntityResolver:
    """Resolves entity variants to canonical forms."""
    
    def __init__(self, config):
        """
        Initialize entity resolver.
        
        Args:
            config: Configuration object
        """
        self.config = config
        # Mapping: variant_text -> canonical_id
        self.variant_to_canonical: Dict[str, str] = {}
        # Mapping: canonical_id -> set of variants
        self.canonical_to_variants: Dict[str, Set[str]] = defaultdict(set)
        # Next canonical ID
        self.next_canonical_id = 0
    
    def resolve(self, variant_text: str) -> str:
        """
        Resolve a variant text to its canonical ID.
        
        Args:
            variant_text: Variant text of the entity
            
        Returns:
            Canonical entity ID
        """
        # Normalize text
        normalized = self._normalize_text(variant_text)
        
        # Check if we've seen this variant
        if normalized in self.variant_to_canonical:
            return self.variant_to_canonical[normalized]
        
        # Check for similar existing entities (fuzzy matching)
        best_match = self._find_best_match(normalized)
        
        if best_match and self._should_merge(normalized, best_match):
            # Merge with existing canonical entity
            canonical_id = self.variant_to_canonical[best_match]
            self._add_variant(normalized, canonical_id)
            return canonical_id
        else:
            # Create new canonical entity
            canonical_id = f"entity_{self.next_canonical_id}"
            self.next_canonical_id += 1
            self._add_variant(normalized, canonical_id)
            return canonical_id
    
    def merge(self, variant_id_1: str, variant_id_2: str, confidence: float) -> str:
        """
        Merge two variant IDs into a single canonical ID.
        
        Args:
            variant_id_1: First variant ID
            variant_id_2: Second variant ID
            confidence: Confidence in the merge
            
        Returns:
            Canonical ID (the one with more variants, or variant_id_1 if equal)
        """
        if confidence < self.config.merge_confidence_threshold:
            return variant_id_1  # Don't merge if confidence too low
        
        # Get canonical IDs
        canon_1 = self.variant_to_canonical.get(variant_id_1, variant_id_1)
        canon_2 = self.variant_to_canonical.get(variant_id_2, variant_id_2)
        
        if canon_1 == canon_2:
            return canon_1  # Already merged
        
        # Choose canonical ID (prefer the one with more variants)
        variants_1 = len(self.canonical_to_variants[canon_1])
        variants_2 = len(self.canonical_to_variants[canon_2])
        
        if variants_2 > variants_1:
            # Merge canon_1 into canon_2
            self._merge_canonicals(canon_1, canon_2)
            return canon_2
        else:
            # Merge canon_2 into canon_1
            self._merge_canonicals(canon_2, canon_1)
            return canon_1
    
    def get_variants(self, canonical_id: str) -> List[str]:
        """
        Get all variant texts for a canonical ID.
        
        Args:
            canonical_id: Canonical entity ID
            
        Returns:
            List of variant texts
        """
        return list(self.canonical_to_variants.get(canonical_id, set()))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase, strip whitespace
        normalized = text.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized
    
    def _find_best_match(self, normalized_text: str) -> Optional[str]:
        """
        Find the best matching existing variant using edit distance.
        
        Args:
            normalized_text: Normalized text to match
            
        Returns:
            Best matching variant text, or None
        """
        best_match = None
        best_distance = float("inf")
        
        for variant in self.variant_to_canonical.keys():
            distance = self._edit_distance(normalized_text, variant)
            if distance < best_distance and distance <= self.config.edit_distance_threshold:
                best_distance = distance
                best_match = variant
        
        return best_match
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein edit distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1],  # substitution
                    )
        
        return dp[m][n]
    
    def _should_merge(self, text1: str, text2: str) -> bool:
        """
        Determine if two texts should be merged.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if texts should be merged
        """
        # Check edit distance
        distance = self._edit_distance(text1, text2)
        if distance <= self.config.edit_distance_threshold:
            return True
        
        # Check for acronym expansion (e.g., "FT" -> "Franklin Templeton")
        if self._is_acronym_expansion(text1, text2):
            return True
        
        return False
    
    def _is_acronym_expansion(self, short: str, long: str) -> bool:
        """
        Check if short text is an acronym of long text.
        
        Args:
            short: Short text (potential acronym)
            long: Long text (potential expansion)
            
        Returns:
            True if short is an acronym of long
        """
        if len(short) < 2 or len(long) < len(short):
            return False
        
        # Extract first letters of words in long text
        long_words = long.split()
        if len(long_words) < len(short):
            return False
        
        acronym = "".join(word[0].upper() for word in long_words[: len(short)])
        return acronym == short.upper()
    
    def _add_variant(self, variant_text: str, canonical_id: str):
        """Add a variant to the mapping."""
        self.variant_to_canonical[variant_text] = canonical_id
        self.canonical_to_variants[canonical_id].add(variant_text)
    
    def _merge_canonicals(self, source_canonical: str, target_canonical: str):
        """Merge source canonical into target canonical."""
        # Move all variants from source to target
        variants = self.canonical_to_variants.pop(source_canonical, set())
        for variant in variants:
            self.variant_to_canonical[variant] = target_canonical
            self.canonical_to_variants[target_canonical].add(variant)

