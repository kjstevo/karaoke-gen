from typing import List, Tuple, Dict, Any, Optional
import logging

from karaoke_gen.lyrics_transcriber.types import GapSequence, WordCorrection
from karaoke_gen.lyrics_transcriber.correction.handlers.base import GapCorrectionHandler
from karaoke_gen.lyrics_transcriber.correction.handlers.word_operations import WordOperations


class FallbackReferenceHandler(GapCorrectionHandler):
    """Fallback handler that always prefers reference lyrics over transcribed lyrics.

    Runs last. When reference lyrics cover a gap (regardless of word count), this
    replaces as many transcribed words as possible with reference words. Any transcribed
    words beyond the reference coverage are left unchanged.

    If no reference source covers the gap at all, the transcription is kept as-is.
    This implements the policy: reference lyrics are authoritative; transcription is
    only used for sections with no reference coverage (ad-libs, unlisted verses, etc.).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.logger = logger or logging.getLogger(__name__)

    def can_handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        if not gap.reference_word_ids:
            return False, {}

        if not self._validate_data(data):
            return False, {}

        # Use whichever source has the most reference words for this gap
        best_source = max(gap.reference_word_ids, key=lambda s: len(gap.reference_word_ids[s]))
        return True, {"word_map": data["word_map"], "source": best_source}

    def handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> List[WordCorrection]:
        if not self._validate_data(data):
            return []

        corrections = []
        word_map = data["word_map"]
        source = data.get("source") or max(
            gap.reference_word_ids, key=lambda s: len(gap.reference_word_ids[s])
        )
        reference_word_ids = gap.reference_word_ids[source]
        reference_positions = WordOperations.calculate_reference_positions(gap)

        ref_count = len(reference_word_ids)
        gap_count = gap.length
        if ref_count != gap_count:
            self.logger.debug(
                f"Word count mismatch: gap has {gap_count} word(s), reference has {ref_count}. "
                f"Replacing {min(gap_count, ref_count)} word(s) from reference."
            )

        for i, (orig_word_id, ref_word_id) in enumerate(zip(gap.transcribed_word_ids, reference_word_ids)):
            if orig_word_id not in word_map:
                self.logger.error(f"Original word ID {orig_word_id} not found in word_map")
                continue
            orig_word = word_map[orig_word_id]

            if ref_word_id not in word_map:
                self.logger.error(f"Reference word ID {ref_word_id} not found in word_map")
                continue
            ref_word = word_map[ref_word_id]

            if orig_word.text.lower() != ref_word.text.lower():
                correction = WordOperations.create_word_replacement_correction(
                    original_word=orig_word.text,
                    corrected_word=ref_word.text,
                    original_position=gap.transcription_position + i,
                    source=source,
                    confidence=0.9,
                    reason="Reference lyrics preferred over transcription; no reference coverage gap",
                    reference_positions=reference_positions,
                    handler="FallbackReferenceHandler",
                    original_word_id=orig_word_id,
                    corrected_word_id=ref_word_id,
                )
                corrections.append(correction)

        return corrections
