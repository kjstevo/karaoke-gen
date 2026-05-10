from typing import List, Tuple, Dict, Any, Optional
import logging

from karaoke_gen.lyrics_transcriber.types import GapSequence, WordCorrection
from karaoke_gen.lyrics_transcriber.correction.handlers.base import GapCorrectionHandler
from karaoke_gen.lyrics_transcriber.correction.handlers.word_operations import WordOperations


class FallbackReferenceHandler(GapCorrectionHandler):
    """Fallback handler that replaces gap words with reference lyrics when word counts match.

    Runs last. When no other handler corrected a gap, this uses the reference text directly
    as long as at least one source has the same word count as the gap. Useful when the
    Whisper transcription is too garbled for other handlers to match but the reference
    word count lines up (e.g. "Cold lucky like pillows" -> "Code Monkey like Fritos").
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.logger = logger or logging.getLogger(__name__)

    def can_handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        if not gap.reference_word_ids:
            return False, {}

        if not self._validate_data(data):
            return False, {}

        # Find any source whose word count matches the gap
        for source, word_ids in gap.reference_word_ids.items():
            if len(word_ids) == gap.length:
                return True, {"word_map": data["word_map"], "source": source}

        self.logger.debug("No reference source has matching word count for fallback.")
        return False, {}

    def handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> List[WordCorrection]:
        if not self._validate_data(data):
            return []

        corrections = []
        word_map = data["word_map"]
        source = data.get("source") or next(
            s for s, ids in gap.reference_word_ids.items() if len(ids) == gap.length
        )
        reference_word_ids = gap.reference_word_ids[source]
        reference_positions = WordOperations.calculate_reference_positions(gap)

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
                    confidence=0.7,
                    reason="Fallback: reference word count matched gap, no other handler succeeded",
                    reference_positions=reference_positions,
                    handler="FallbackReferenceHandler",
                    original_word_id=orig_word_id,
                    corrected_word_id=ref_word_id,
                )
                corrections.append(correction)

        return corrections
