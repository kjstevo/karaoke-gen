import logging
import re
from typing import Any, Dict, List, Optional

from karaoke_gen.lyrics_transcriber.types import LyricsData, LyricsSegment, Word
from karaoke_gen.lyrics_transcriber.utils.word_utils import WordUtils


def resegment_by_reference(
    corrected_segments: List[LyricsSegment],
    reference_lyrics: Dict[str, LyricsData],
    anchor_sequences: List[Any],
    gap_sequences: Optional[List[Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[LyricsSegment]:
    """Re-segment corrected lyrics to follow reference line boundaries.

    Primary strategy: use anchor sequences (exact word-to-line mapping) and
    gap sequences (proportional reference-word distribution) to assign each
    corrected word to the correct reference line without depending on timing.
    Fallback: time-based assignment when reference has synced timestamps.
    Returns original segments unchanged if no improvement is found.
    """
    log = logger or logging.getLogger(__name__)

    all_words = [w for seg in corrected_segments for w in seg.words]
    if not all_words or not reference_lyrics:
        return corrected_segments

    # Prefer synced (timed) reference sources; otherwise use any available
    ref_data: Optional[LyricsData] = None
    for data in reference_lyrics.values():
        if data.metadata.is_synced and data.segments:
            ref_data = data
            break
    if ref_data is None:
        ref_data = next(iter(reference_lyrics.values()), None)

    if not ref_data or not ref_data.segments:
        return corrected_segments

    ref_segs = ref_data.segments

    # Primary: anchor + gap assignment (no timing at anchors; time-based for large gaps)
    result = _resegment_by_anchor_and_gaps(
        all_words, ref_segs, anchor_sequences, gap_sequences or [], ref_data.source, log,
        ref_is_synced=ref_data.metadata.is_synced,
    )
    if result:
        log.info(f"Reference resegmentation (anchor+gap): {len(corrected_segments)} → {len(result)} segments")
        return result

    # Fallback: time-based (requires synced reference with line timestamps)
    if ref_data.metadata.is_synced:
        result = _resegment_by_time(all_words, ref_segs, log)
        if result:
            log.info(f"Reference resegmentation (time-based): {len(corrected_segments)} → {len(result)} segments")
            return result

    log.debug("Reference resegmentation: no improvement found, keeping original segments")
    return corrected_segments


def _resegment_by_anchor_and_gaps(
    all_words: List[Word],
    ref_segments: List[LyricsSegment],
    anchor_sequences: List[Any],
    gap_sequences: List[Any],
    source: str,
    logger: logging.Logger,
    ref_is_synced: bool = False,
) -> Optional[List[LyricsSegment]]:
    """Assign words using exact anchor mappings and gap interpolation.

    Anchor words: exact 1:1 trans↔ref mapping, no timestamps used.
    Gap words: time-based when the gap spans many reference lines (avoids the
      sparse-proportional problem); proportional otherwise.
    Non-monotonic anchors (repeat-section detection artefacts that point back to
    earlier reference positions) are filtered out to prevent scrambled output.
    """
    # Map reference word ID → segment index
    ref_word_to_seg: Dict[str, int] = {}
    for seg_idx, seg in enumerate(ref_segments):
        for word in seg.words:
            ref_word_to_seg[word.id] = seg_idx

    # Map transcribed word ID → position index in all_words
    trans_id_to_idx: Dict[str, int] = {w.id: i for i, w in enumerate(all_words)}

    assignments: List[Optional[int]] = [None] * len(all_words)

    # Assign anchor words exactly (1:1 trans↔ref mapping).
    # Process in transcription order and skip non-monotonic anchors — these arise
    # when repeat-section detection maps a later transcription position back to an
    # earlier reference position (e.g. a chorus occurrence matched to the first
    # chorus slot in the reference), producing scrambled output.
    sorted_anchors = sorted(
        anchor_sequences,
        key=lambda item: (item.anchor if hasattr(item, "anchor") else item).transcription_position,
    )
    last_max_ref_seg = -1
    for item in sorted_anchors:
        anchor = item.anchor if hasattr(item, "anchor") else item
        if source not in anchor.reference_word_ids:
            continue
        ref_ids = anchor.reference_word_ids[source]
        valid_segs = [ref_word_to_seg[rid] for rid in ref_ids if rid in ref_word_to_seg]
        if not valid_segs:
            continue
        if min(valid_segs) < last_max_ref_seg:
            logger.debug(
                f"Skipping non-monotonic anchor at trans_pos={anchor.transcription_position} "
                f"(min_ref_seg={min(valid_segs)} < prev_max={last_max_ref_seg})"
            )
            continue
        last_max_ref_seg = max(valid_segs)
        for trans_id, ref_id in zip(anchor.transcribed_word_ids, ref_ids):
            idx = trans_id_to_idx.get(trans_id)
            seg_idx = ref_word_to_seg.get(ref_id)
            if idx is not None and seg_idx is not None:
                assignments[idx] = seg_idx

    if not any(a is not None for a in assignments):
        return None

    # Build time boundaries for ref segments (used for synced-reference gap assignment)
    if ref_is_synced:
        boundaries = [
            (seg.start_time, ref_segments[i + 1].start_time if i + 1 < len(ref_segments) else float("inf"))
            for i, seg in enumerate(ref_segments)
        ]
    else:
        boundaries = []

    # Assign gap words.
    # Use time-based assignment when the reference is synced and the gap spans
    # more than 3 reference lines — proportional produces too few output lines when
    # a handful of transcribed words must cover many reference lines.
    # Use proportional assignment for small gaps or unsynced references.
    for gap in gap_sequences:
        if source not in gap.reference_word_ids:
            continue
        ref_ids = gap.reference_word_ids[source]
        if not ref_ids:
            continue
        gap_ref_segs = sorted({ref_word_to_seg[rid] for rid in ref_ids if rid in ref_word_to_seg})
        if not gap_ref_segs:
            continue

        use_time_based = ref_is_synced and len(gap_ref_segs) > 3
        gap_trans_ids = gap.transcribed_word_ids
        N = len(gap_trans_ids)
        M = len(ref_ids)

        if use_time_based:
            lo_seg = gap_ref_segs[0]
            hi_seg = gap_ref_segs[-1]
            for trans_id in gap_trans_ids:
                idx = trans_id_to_idx.get(trans_id)
                if idx is None:
                    continue
                t = all_words[idx].start_time
                assigned = None
                for seg_i in range(lo_seg, hi_seg + 1):
                    lo, hi = boundaries[seg_i]
                    if lo <= t < hi:
                        assigned = seg_i
                        break
                if assigned is None:
                    # Clamp to nearest boundary in the gap range
                    assigned = min(
                        range(lo_seg, hi_seg + 1),
                        key=lambda i: min(abs(t - boundaries[i][0]), abs(t - boundaries[i][1])),
                    )
                assignments[idx] = assigned
        else:
            for i, trans_id in enumerate(gap_trans_ids):
                idx = trans_id_to_idx.get(trans_id)
                if idx is None:
                    continue
                ref_pos = round(i / (N - 1) * (M - 1)) if N > 1 else 0
                ref_id = ref_ids[min(ref_pos, M - 1)]
                seg_idx = ref_word_to_seg.get(ref_id)
                if seg_idx is not None:
                    assignments[idx] = seg_idx

    # For synced references, time-assign any word still unassigned (e.g. words from
    # dropped non-monotonic anchors, or words in regions with no gap coverage).
    # This avoids aggressive forward-fill merging large blocks into one line.
    if ref_is_synced:
        for i, word in enumerate(all_words):
            if assignments[i] is not None:
                continue
            t = word.start_time
            for j, (lo, hi) in enumerate(boundaries):
                if lo <= t < hi:
                    assignments[i] = j
                    break

    # Forward-backward fill for any remaining unassigned words (outside all time windows)
    _forward_backward_fill(assignments)

    groups: Dict[int, List[Word]] = {}
    for word, seg_idx in zip(all_words, assignments):
        groups.setdefault(seg_idx or 0, []).append(word)

    if len(groups) <= 1:
        return None

    result = []
    for seg_idx in sorted(groups.keys()):
        words = groups[seg_idx]
        result.append(
            LyricsSegment(
                id=WordUtils.generate_id(),
                text=" ".join(w.text for w in words),
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            )
        )
    return result


def _resegment_by_time(
    all_words: List[Word],
    ref_segments: List[LyricsSegment],
    logger: logging.Logger,
) -> Optional[List[LyricsSegment]]:
    """Assign words to reference segment slots based on word start_time vs reference line timestamps."""
    # Build half-open intervals [line_start, next_line_start) for each reference segment
    boundaries = []
    for i, seg in enumerate(ref_segments):
        lo = seg.start_time
        hi = ref_segments[i + 1].start_time if i + 1 < len(ref_segments) else float("inf")
        boundaries.append((lo, hi))

    groups: Dict[int, List[Word]] = {}
    for word in all_words:
        t = word.start_time
        assigned = None
        for i, (lo, hi) in enumerate(boundaries):
            if lo <= t < hi:
                assigned = i
                break
        if assigned is None:
            # Before first line → first segment; after last → last segment
            assigned = 0 if t < boundaries[0][0] else len(ref_segments) - 1
        groups.setdefault(assigned, []).append(word)

    if len(groups) <= 1:
        return None

    result = []
    for seg_idx in sorted(groups.keys()):
        words = groups[seg_idx]
        result.append(
            LyricsSegment(
                id=WordUtils.generate_id(),
                text=" ".join(w.text for w in words),
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            )
        )
    return result


def _resegment_by_anchors(
    all_words: List[Word],
    ref_segments: List[LyricsSegment],
    anchor_sequences: List[Any],
    source: str,
    logger: logging.Logger,
) -> Optional[List[LyricsSegment]]:
    """Assign words to reference segment slots using anchor word-ID mapping, interpolating gaps."""
    # Map reference word ID → segment index
    ref_word_to_seg: Dict[str, int] = {}
    for seg_idx, seg in enumerate(ref_segments):
        for word in seg.words:
            ref_word_to_seg[word.id] = seg_idx

    # Map transcribed word ID → reference word ID via anchors
    # Each item may be AnchorSequence or ScoredAnchor (has .anchor attribute)
    trans_to_ref: Dict[str, str] = {}
    for item in anchor_sequences:
        anchor = item.anchor if hasattr(item, "anchor") else item
        if source not in anchor.reference_word_ids:
            continue
        for trans_id, ref_id in zip(anchor.transcribed_word_ids, anchor.reference_word_ids[source]):
            trans_to_ref[trans_id] = ref_id

    # Assign known anchor words; leave gap words as None
    assignments: List[Optional[int]] = [None] * len(all_words)
    for i, word in enumerate(all_words):
        ref_id = trans_to_ref.get(word.id)
        if ref_id and ref_id in ref_word_to_seg:
            assignments[i] = ref_word_to_seg[ref_id]

    if not any(a is not None for a in assignments):
        return None

    # Forward fill then backward fill to cover gap words
    _forward_backward_fill(assignments)

    # Group words by assigned reference segment
    groups: Dict[int, List[Word]] = {}
    for word, seg_idx in zip(all_words, assignments):
        groups.setdefault(seg_idx or 0, []).append(word)

    if len(groups) <= 1:
        return None

    result = []
    for seg_idx in sorted(groups.keys()):
        words = groups[seg_idx]
        result.append(
            LyricsSegment(
                id=WordUtils.generate_id(),
                text=" ".join(w.text for w in words),
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            )
        )
    return result


def _forward_backward_fill(assignments: List[Optional[int]]) -> None:
    """Fill None entries by propagating nearest known values (forward then backward)."""
    last: Optional[int] = None
    for i in range(len(assignments)):
        if assignments[i] is not None:
            last = assignments[i]
        elif last is not None:
            assignments[i] = last
    last = None
    for i in range(len(assignments) - 1, -1, -1):
        if assignments[i] is not None:
            last = assignments[i]
        elif last is not None:
            assignments[i] = last


class SegmentResizer:
    """Handles resizing of lyrics segments to ensure proper line lengths and natural breaks.

    This class processes lyrics segments and splits them into smaller segments when they exceed
    a maximum line length. It attempts to split at natural break points like sentence endings,
    commas, or conjunctions to maintain readability.

    Example:
        resizer = SegmentResizer(max_line_length=36)
        segments = [
            LyricsSegment(
                text="This is a very long sentence that needs to be split into multiple lines for better readability",
                words=[...],  # List of Word objects with timing information
                start_time=0.0,
                end_time=5.0
            )
        ]
        resized = resizer.resize_segments(segments)
        # Results in:
        # [
        #     LyricsSegment(text="This is a very long sentence", ...),
        #     LyricsSegment(text="that needs to be split", ...),
        #     LyricsSegment(text="into multiple lines", ...),
        #     LyricsSegment(text="for better readability", ...)
        # ]
    """

    def __init__(self, max_line_length: int = 36, logger: Optional[logging.Logger] = None):
        """Initialize the SegmentResizer.

        Args:
            max_line_length: Maximum allowed length for a single line of text
            logger: Optional logger for debugging information
        """
        self.max_line_length = max_line_length
        self.logger = logger or logging.getLogger(__name__)

    def resize_segments(self, segments: List[LyricsSegment]) -> List[LyricsSegment]:
        """Main entry point for resizing segments.

        Takes a list of potentially long segments and splits them into smaller ones
        while preserving word timing information.

        Example:
            Input segment: "Hello world, this is a test. And here's another sentence."
            Output segments: [
                "Hello world, this is a test.",
                "And here's another sentence."
            ]

        Args:
            segments: List of LyricsSegment objects to process

        Returns:
            List of resized LyricsSegment objects
        """
        self._log_input_segments(segments)
        resized_segments: List[LyricsSegment] = []

        for segment_idx, segment in enumerate(segments):
            cleaned_segment = self._create_cleaned_segment(segment)

            # Only split if the segment is longer than max_line_length
            if len(cleaned_segment.text) <= self.max_line_length:
                resized_segments.append(cleaned_segment)
                continue

            # Process oversized segments
            resized_segments.extend(self._split_oversized_segment(segment_idx, segment))

        self._log_output_segments(resized_segments)
        return resized_segments

    def _clean_text(self, text: str) -> str:
        """Clean text by removing newlines and extra whitespace.

        Example:
            Input: "Hello\n  World  \n!"
            Output: "Hello World !"

        Args:
            text: String to clean

        Returns:
            Cleaned string with normalized whitespace
        """
        return " ".join(text.replace("\n", " ").split())

    def _create_cleaned_segment(self, segment: LyricsSegment) -> LyricsSegment:
        """Create a new segment with cleaned text while preserving timing info.

        Example:
            Input: LyricsSegment(text="Hello\n  World\n", words=[...])
            Output: LyricsSegment(text="Hello World", words=[...])
        """
        cleaned_text = self._clean_text(segment.text)
        return LyricsSegment(
            id=segment.id,  # Preserve the original segment ID
            text=cleaned_text,
            words=segment.words,
            start_time=segment.start_time,
            end_time=segment.end_time,
            singer=segment.singer,
        )

    def _create_cleaned_word(self, word: Word) -> Word:
        """Create a new word with cleaned text."""
        cleaned_text = self._clean_text(word.text)
        return Word(
            id=word.id,  # Preserve the original word ID
            text=cleaned_text,
            start_time=word.start_time,
            end_time=word.end_time,
            confidence=word.confidence if hasattr(word, "confidence") else None,
            created_during_correction=getattr(word, "created_during_correction", False),
            singer=word.singer,
        )

    def _split_oversized_segment(self, segment_idx: int, segment: LyricsSegment) -> List[LyricsSegment]:
        """Split an oversized segment into multiple segments at natural break points.

        Example:
            Input: "This is a long sentence. Here's another one."
            Output: [
                LyricsSegment(text="This is a long sentence.", ...),
                LyricsSegment(text="Here's another one.", ...)
            ]
        """
        segment_text = self._clean_text(segment.text)

        self.logger.info(f"Processing oversized segment {segment_idx}: '{segment_text}'")
        split_lines = self._process_segment_text(segment_text)
        self.logger.debug(f"Split into {len(split_lines)} lines: {split_lines}")

        return self._create_segments_from_lines(segment_text, split_lines, segment.words, singer=segment.singer)

    def _create_segments_from_lines(
        self,
        segment_text: str,
        split_lines: List[str],
        words: List[Word],
        singer: Optional[int] = None,
    ) -> List[LyricsSegment]:
        """Create segments from split lines while preserving word timing.

        Matches words to their corresponding lines based on text position and
        creates new segments with the correct timing information.

        Example:
            segment_text: "Hello world, how are you"
            split_lines: ["Hello world,", "how are you"]
            words: [Word("Hello", 0.0, 1.0), Word("world", 1.0, 2.0), ...]

        Returns segments with words properly assigned to each line.
        """
        segments: List[LyricsSegment] = []
        words_to_process = words.copy()
        current_pos = 0

        for line in split_lines:
            line_words = []
            line_text = line.strip()
            remaining_line = line_text

            # Keep processing words until we've found all words for this line
            while words_to_process and remaining_line:
                word = words_to_process[0]
                word_clean = self._clean_text(word.text)

                # Check if the cleaned word appears in the remaining line text
                if word_clean in remaining_line:
                    word_pos = remaining_line.find(word_clean)
                    if word_pos != -1:
                        line_words.append(words_to_process.pop(0))
                        # Remove the word and any following spaces from remaining line
                        remaining_line = remaining_line[word_pos + len(word_clean) :].strip()
                        continue

                # If we can't find the word in the remaining line, we're done with this line
                break

            if line_words:
                segments.append(self._create_segment_from_words(line, line_words, singer=singer))
                current_pos += len(line) + 1  # +1 for the space between lines

        # If we have any remaining words, create a final segment with them
        if words_to_process:
            remaining_text = " ".join(self._clean_text(w.text) for w in words_to_process)
            segments.append(self._create_segment_from_words(remaining_text, words_to_process, singer=singer))

        return segments

    def _create_line_segment(
        self, line_idx: int, line: str, segment_text: str, available_words: List[Word], current_pos: int
    ) -> Optional[LyricsSegment]:
        """Create a single segment from a line of text."""
        line_pos = segment_text.find(line, current_pos)
        if line_pos == -1:
            self.logger.error(f"Failed to find line '{line}' in segment text '{segment_text}' " f"starting from position {current_pos}")
            return None

        line_words = self._find_words_for_line(line, line_pos, len(line), segment_text, available_words, current_pos)

        if line_words:
            return self._create_segment_from_words(line, line_words)
        else:
            self.logger.warning(f"No words found for line '{line}'")
            return None

    def _find_words_for_line(
        self, line: str, line_pos: int, line_length: int, segment_text: str, available_words: List[Word], current_pos: int
    ) -> List[Word]:
        """Find words that belong to a specific line."""
        line_words = []
        line_text = line.strip()
        remaining_text = line_text

        for word in available_words:
            # Skip if word isn't in remaining text
            if word.text not in remaining_text:
                continue

            # Find position of word in line
            word_pos = remaining_text.find(word.text)
            if word_pos != -1:
                line_words.append(word)
                # Remove processed text up to and including this word
                remaining_text = remaining_text[word_pos + len(word.text) :].strip()

            if not remaining_text:  # All words found
                break

        return line_words

    def _create_segment_from_words(
        self,
        line: str,
        words: List[Word],
        singer: Optional[int] = None,
    ) -> LyricsSegment:
        """Create a new segment from a list of words."""
        cleaned_text = self._clean_text(line)
        return LyricsSegment(
            id=WordUtils.generate_id(),  # Generate new ID for split segments
            text=cleaned_text,
            words=words,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            singer=singer,
        )

    def _process_segment_text(self, text: str) -> List[str]:
        """Process segment text to determine optimal split points."""
        self.logger.debug(f"Processing segment text: '{text}'")
        processed_lines: List[str] = []
        remaining_text = text.strip()

        while remaining_text:
            self.logger.debug(f"Remaining text to process: '{remaining_text}'")

            # If remaining text is within limit, add it and we're done
            if len(remaining_text) <= self.max_line_length:
                processed_lines.append(remaining_text)
                break

            # Find best split point
            split_point = self._find_best_split_point(remaining_text)
            first_part = remaining_text[:split_point].strip()
            second_part = remaining_text[split_point:].strip()

            # Only split if:
            # 1. We found a valid split point
            # 2. First part isn't too long
            # 3. Both parts are non-empty
            if split_point < len(remaining_text) and len(first_part) <= self.max_line_length and first_part and second_part:

                processed_lines.append(first_part)
                remaining_text = second_part
            else:
                # If we can't find a good split, keep the whole text
                processed_lines.append(remaining_text)
                break

        return processed_lines

    def _find_best_split_point(self, line: str) -> int:
        """Find the best split point that creates natural, well-balanced segments."""
        self.logger.debug(f"Finding best split point for line: '{line}' (length: {len(line)})")

        # If line is within max length, don't split
        if len(line) <= self.max_line_length:
            return len(line)

        break_points = self._find_break_points(line)
        best_point = None
        best_score = float("-inf")

        # Try each break point and score it
        for priority, points in enumerate(break_points):
            for point in sorted(points):  # Sort points to prefer earlier ones in same priority
                if point <= 0 or point >= len(line):
                    continue

                first_part = line[:point].strip()

                # Skip if first part is too long
                if len(first_part) > self.max_line_length:
                    continue

                # Score this break point
                score = self._score_break_point(line, point, priority)
                if score > best_score:
                    best_score = score
                    best_point = point

        # If no good break points found, fall back to last space before max_length
        if best_point is None:
            last_space = line.rfind(" ", 0, self.max_line_length)
            if last_space != -1:
                return last_space

        return best_point if best_point is not None else self.max_line_length

    def _score_break_point(self, line: str, point: int, priority: int) -> float:
        """Score a potential break point based on multiple factors.

        Factors considered:
        1. Priority of the break point type (sentence > clause > comma, etc.)
        2. Balance of segment lengths
        3. Proximity to target length

        Example:
            line: "This is a sentence. And more text."
            point: 18 (after "sentence.")
            priority: 0 (sentence break)

        Returns a score where higher is better. Score components:
        - Base score (100-20*priority): 100 for priority 0
        - Length ratio bonus (0-10): Based on segment balance
        - Target length bonus (0-5): Based on proximity to ideal length
        """
        first_segment = line[:point].strip()
        second_segment = line[point:].strip()

        # Base score starts with priority
        score = 100 - (priority * 20)  # Priorities 0-4 give scores 100,80,60,40,20

        # Length ratio bonus
        length_ratio = min(len(first_segment), len(second_segment)) / max(len(first_segment), len(second_segment))
        score += length_ratio * 10

        # Target length bonus
        target_length = self.max_line_length * 0.7
        first_length_score = 1 - abs(len(first_segment) - target_length) / self.max_line_length
        score += first_length_score * 5

        return score

    def _find_break_points(self, line: str) -> List[List[int]]:
        """Find potential break points in order of preference.

        Returns a list of lists, where each inner list contains break points
        of the same priority. Break points are indices where text should be split.

        Priority order:
        1. Sentence endings (., !, ?)
        2. Major clause breaks (;, -)
        3. Comma breaks
        4. Coordinating conjunctions (and, but, or)
        5. Prepositions/articles (in, at, the, a)

        Example:
            Input: "Hello, world. This is a test"
            Output: [
                [12],  # sentence break after "world."
                [],   # no semicolons or dashes
                [5],  # comma after "Hello,"
                [],   # no conjunctions
                [15]  # preposition "is"
            ]
        """
        break_points = []

        # Priority 1: Sentence endings
        sentence_breaks = []
        for punct in [".", "!", "?"]:
            for match in re.finditer(rf"\{punct}\s+", line):
                sentence_breaks.append(match.start() + 1)
        break_points.append(sentence_breaks)

        # Priority 2: Major clause breaks (semicolons, dashes)
        major_breaks = []
        for punct in [";", " - "]:
            for match in re.finditer(re.escape(punct), line):
                major_breaks.append(match.start())  # Position before the punctuation
        break_points.append(major_breaks)

        # Priority 3: Comma breaks, typically marking natural pauses
        comma_breaks = []
        for match in re.finditer(r",\s+", line):
            comma_breaks.append(match.start() + 1)  # Position after the comma
        break_points.append(comma_breaks)

        # Priority 4: Coordinating conjunctions with surrounding spaces
        conjunction_breaks = []
        for conj in [" and ", " but ", " or "]:
            for match in re.finditer(re.escape(conj), line):
                conjunction_breaks.append(match.start())  # Position before the conjunction
        break_points.append(conjunction_breaks)

        # Priority 5: Prepositions or articles with surrounding spaces (last resort)
        minor_breaks = []
        for prep in [" in ", " at ", " the ", " a "]:
            for match in re.finditer(re.escape(prep), line):
                minor_breaks.append(match.start())  # Position before the preposition
        break_points.append(minor_breaks)

        return break_points

    def _log_input_segments(self, segments: List[LyricsSegment]) -> None:
        """Log input segment information."""
        self.logger.info(f"Starting segment resize. Input: {len(segments)} segments")
        for idx, segment in enumerate(segments):
            self.logger.debug(
                f"Input segment {idx}: text='{segment.text}', "
                f"words={len(segment.words)} words, "
                f"time={segment.start_time:.2f}-{segment.end_time:.2f}"
            )

    def _log_output_segments(self, segments: List[LyricsSegment]) -> None:
        """Log output segment information."""
        self.logger.info(f"Finished resizing. Output: {len(segments)} segments")
        for idx, segment in enumerate(segments):
            self.logger.debug(
                f"Output segment {idx}: text='{segment.text}', "
                f"words={len(segment.words)} words, "
                f"time={segment.start_time:.2f}-{segment.end_time:.2f}"
            )
