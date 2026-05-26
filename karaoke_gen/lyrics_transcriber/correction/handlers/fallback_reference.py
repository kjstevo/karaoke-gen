from typing import List, Tuple, Dict, Any, Optional
import logging

from karaoke_gen.lyrics_transcriber.types import GapSequence, WordCorrection
from karaoke_gen.lyrics_transcriber.correction.handlers.base import GapCorrectionHandler
from karaoke_gen.lyrics_transcriber.correction.handlers.word_operations import WordOperations


class FallbackReferenceHandler(GapCorrectionHandler):
    """Fallback handler that always prefers reference lyrics over transcribed lyrics.

    Runs last. For any gap that has reference coverage (either pre-assigned by the
    anchor finder, or inferred from the preceding anchor's reference position), this
    replaces transcribed words with reference words. Any transcribed words beyond the
    reference coverage are left unchanged.

    If no reference source covers the gap at all, the transcription is kept as-is.
    This implements the policy: reference lyrics are authoritative; transcription is
    only used for sections with no reference coverage (ad-libs, unlisted verses, etc.).

    Handles the common "orphaned gap" case in repetitive songs (e.g. a phrase that
    appears 15 times but the anchor finder only claimed reference positions for some
    instances). When gap.reference_word_ids is empty, the handler looks at where the
    preceding anchor ends in the reference and takes the next N words from there.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.logger = logger or logging.getLogger(__name__)

    def _infer_reference_words(
        self,
        gap: GapSequence,
        anchor_sequences: list,
        reference_lyrics: dict,
        word_map: dict = None,
    ) -> Dict[str, List[str]]:
        """Infer reference word IDs for a gap from the preceding anchor's reference position.

        Finds where the preceding anchor's last reference word sits in the ordered
        reference word list, then returns the next gap.length words from that position.
        Works even when gap.reference_word_ids is empty due to anchor position collisions
        in repetitive songs.
        """
        if not gap.preceding_anchor_id:
            self.logger.info("FallbackRef inference: no preceding_anchor_id on gap — cannot infer")
            return {}

        # anchor_sequences may contain ScoredAnchor wrappers — unwrap to AnchorSequence
        anchors = [a.anchor if hasattr(a, "anchor") else a for a in anchor_sequences]
        anchor_by_id = {a.id: a for a in anchors}
        preceding_anchor = anchor_by_id.get(gap.preceding_anchor_id)
        if not preceding_anchor:
            self.logger.info(
                f"FallbackRef inference: preceding_anchor_id={gap.preceding_anchor_id} not found in "
                f"{len(anchors)} anchors — cannot infer"
            )
            return {}

        result = {}
        for source, lyrics_data in reference_lyrics.items():
            anchor_ref_ids = preceding_anchor.reference_word_ids.get(source, [])
            if not anchor_ref_ids:
                self.logger.info(
                    f"FallbackRef inference: preceding anchor has no ref IDs for source={source} — skipping"
                )
                continue

            # Build ordered reference word ID list for this source
            all_ref_word_ids = [w.id for seg in lyrics_data.segments for w in seg.words]
            if not all_ref_word_ids:
                self.logger.info(f"FallbackRef inference: source={source} has no words — skipping")
                continue

            # Find where the preceding anchor ends in the reference
            last_anchor_ref_id = anchor_ref_ids[-1]
            if last_anchor_ref_id not in all_ref_word_ids:
                # Cross-session ID mismatch: anchor cached with different reference word IDs
                # Fall back to reference_positions (clean-text index) if available
                anchor_ref_pos = preceding_anchor.reference_positions.get(source)
                anchor_ref_len = len(anchor_ref_ids)
                if anchor_ref_pos is not None:
                    anchor_end_pos = anchor_ref_pos + anchor_ref_len
                    self.logger.info(
                        f"FallbackRef inference: anchor ref ID not in reference list (stale cache?) "
                        f"for source={source}; using reference_positions fallback: "
                        f"anchor_ref_pos={anchor_ref_pos}, anchor_ref_len={anchor_ref_len}, "
                        f"anchor_end_pos={anchor_end_pos}"
                    )
                else:
                    self.logger.info(
                        f"FallbackRef inference: anchor ref ID not in reference list for source={source} "
                        f"and no reference_positions available — cannot infer"
                    )
                    continue
            else:
                try:
                    anchor_end_pos = all_ref_word_ids.index(last_anchor_ref_id) + 1
                except ValueError:
                    self.logger.info(
                        f"FallbackRef inference: ValueError finding anchor ref ID in reference "
                        f"word list for source={source} — skipping"
                    )
                    continue

            # Take the next gap.length words from the reference
            candidate_ids = all_ref_word_ids[anchor_end_pos: anchor_end_pos + gap.length]
            if candidate_ids:
                candidate_texts = [word_map[wid].text if word_map and wid in word_map else wid for wid in candidate_ids]
                result[source] = candidate_ids
                self.logger.info(
                    f"FallbackRef inference: inferred {len(candidate_ids)} word(s) for source={source} "
                    f"at ref pos {anchor_end_pos}: {candidate_texts}"
                )
            else:
                self.logger.info(
                    f"FallbackRef inference: anchor ends at ref pos {anchor_end_pos} but no more words "
                    f"in reference for source={source} (ref has {len(all_ref_word_ids)} total words)"
                )

        return result

    def can_handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        if not self._validate_data(data):
            return False, {}

        word_map = data["word_map"]

        # Check whether reference_word_ids has any non-empty lists
        has_ref_words = gap.reference_word_ids and any(ids for ids in gap.reference_word_ids.values())

        if has_ref_words:
            best_source = max(gap.reference_word_ids, key=lambda s: len(gap.reference_word_ids[s]))
            ref_count = len(gap.reference_word_ids[best_source])
            self.logger.debug(
                f"FallbackRef: gap has {ref_count} direct ref word(s) via source={best_source}"
            )
            return True, {
                "word_map": word_map,
                "source": best_source,
                "effective_reference_word_ids": gap.reference_word_ids,
            }

        # Gap has no pre-assigned reference words — try to infer from preceding anchor position
        anchor_sequences = data.get("anchor_sequences", [])
        reference_lyrics = data.get("reference_lyrics", {})

        if not anchor_sequences or not reference_lyrics:
            return False, {}

        # Resolve gap text for logging
        gap_text = " ".join(word_map[wid].text for wid in gap.transcribed_word_ids if wid in word_map)
        self.logger.info(
            f"FallbackRef: gap '{gap_text}' has no direct ref words — attempting position inference"
        )

        inferred = self._infer_reference_words(gap, anchor_sequences, reference_lyrics, word_map=word_map)
        if not inferred:
            self.logger.info(f"FallbackRef: inference failed for gap '{gap_text}' — leaving uncorrected")
            return False, {}

        best_source = max(inferred, key=lambda s: len(inferred[s]))
        return True, {
            "word_map": word_map,
            "source": best_source,
            "effective_reference_word_ids": inferred,
        }

    def handle(self, gap: GapSequence, data: Optional[Dict[str, Any]] = None) -> List[WordCorrection]:
        if not self._validate_data(data):
            return []

        corrections = []
        word_map = data["word_map"]
        effective_ref = data.get("effective_reference_word_ids") or gap.reference_word_ids
        source = data.get("source") or max(effective_ref, key=lambda s: len(effective_ref[s]))
        reference_word_ids = effective_ref.get(source, [])
        reference_positions = WordOperations.calculate_reference_positions(gap)

        ref_count = len(reference_word_ids)
        gap_count = gap.length
        if ref_count != gap_count:
            self.logger.debug(
                f"FallbackRef: word count mismatch — gap has {gap_count} word(s), "
                f"reference has {ref_count}; will replace {min(gap_count, ref_count)}"
            )

        for i, (orig_word_id, ref_word_id) in enumerate(zip(gap.transcribed_word_ids, reference_word_ids)):
            if orig_word_id not in word_map:
                self.logger.error(f"FallbackRef: original word ID {orig_word_id} not in word_map")
                continue
            orig_word = word_map[orig_word_id]

            if ref_word_id not in word_map:
                self.logger.error(
                    f"FallbackRef: reference word ID {ref_word_id} (source={source}) not in word_map "
                    f"— this indicates a stale anchor cache; delete the cache and reprocess"
                )
                continue
            ref_word = word_map[ref_word_id]

            if orig_word.text.lower() != ref_word.text.lower():
                self.logger.info(
                    f"FallbackRef: correcting '{orig_word.text}' → '{ref_word.text}' (source={source})"
                )
                correction = WordOperations.create_word_replacement_correction(
                    original_word=orig_word.text,
                    corrected_word=ref_word.text,
                    original_position=gap.transcription_position + i,
                    source=source,
                    confidence=0.9,
                    reason="Reference lyrics preferred over transcription",
                    reference_positions=reference_positions,
                    handler="FallbackReferenceHandler",
                    original_word_id=orig_word_id,
                    corrected_word_id=ref_word_id,
                )
                corrections.append(correction)

        return corrections
