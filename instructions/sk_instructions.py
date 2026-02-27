# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of Slovak instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

from absl import logging
import langdetect
import unicodedata

from instruction_utils import sk_instructions_util


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = sk_instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
# "aspoň" = at least, "menej ako" = less than / at most.
# "minimálne" is accepted as a synonym for "aspoň" (see _normalize_relation).
_COMPARISON_RELATION = ("aspoň", "menej ako")

# Maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# Number of placeholders.
_NUM_PLACEHOLDERS = 4

# Number of bullet lists.
_NUM_BULLETS = 5

# Options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = ("Áno.", "Nie.", "Možno.")

# Options of starter keywords.
_STARTER_OPTIONS = ("Povedal by som", "Moja odpoveď je", "Myslím si",
                    "Podľa môjho názoru", "Z môjho pohľadu",
                    "Ten", "Tá", "To", "Jeden", "Jedna", "Pre", "Za",
                    "Potom", "Ona", "On")

# Options of ending keywords.
_ENDING_OPTIONS = ("S pozdravom", "Potrebujete niečo ďalšie?",
                   "Čakám na ďalšie otázky")

# Number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# Section splitter keywords.
_SECTION_SPLITER = ("Sekcia", "SEKCIA")

# Number of sections.
_NUM_SECTIONS = 5

# Number of paragraphs.
_NUM_PARAGRAPHS = 5

# Postscript markers used in Slovak.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# Number of keywords.
_NUM_KEYWORDS = 2

# Occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# Occurrences of a single letter.
_LETTER_FREQUENCY = 20

# Occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# Number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 1
_NUM_WORDS_UPPER_LIMIT = 500


def _normalize_relation(relation):
  """Normalises Slovak relation synonyms to a canonical form."""
  if relation == "minimálne":
    return "aspoň"
  return relation


class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
  """Check the language of the entire response."""

  def build_description(self, *, language=None):
    """Build the instruction description.

    Args:
      language: A string representing the expected language of the response
        following ISO 639-1 codes.

    Returns:
      A string representing the instruction description.
    """
    self._language = language
    if self._language is None:
      self._language = random.choice(list(_LANGUAGES.keys()))
    self._description_pattern = (
        "Celá tvoja odpoveď musí byť v jazyku {language}, "
        "žiadny iný jazyk nie je povolený.")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    return {"language": self._language}

  def get_instruction_args_keys(self):
    return ["language"]

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return langdetect.detect(value) == self._language
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e)
      return True


class NumberOfSentences(Instruction):
  """Check the number of sentences."""

  def build_description(self, *, num_sentences=None, relation=None):
    """Build the instruction description.

    Args:
      num_sentences: An integer specifying the number of sentences as a
        threshold.
      relation: A string in _COMPARISON_RELATION (or "minimálne" as alias).

    Returns:
      A string representing the instruction description.
    """
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or
        self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    relation = _normalize_relation(relation)
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Tvoja odpoveď musí obsahovať {relation} {num_sentences} viet.")
    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["num_sentences", "relation"]

  def check_following(self, value):
    cleaned_text = re.sub(r'(^\s*[\d]+\.\s*)|(^\s*[-*]\s*)', '', value,
                          flags=re.MULTILINE)
    num_sentences = sk_instructions_util.count_sentences(cleaned_text)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_sentences >= self._num_sentences_threshold
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_sentences <= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders=None, relation=None):
    """Build the instruction description.

    Args:
      num_placeholders: An integer denoting the minimum number of placeholders.
      relation: Unused; placeholders always use "at least" semantics.

    Returns:
      A string representing the instruction description.
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)

    self._description_pattern = (
        "Odpoveď musí obsahovať aspoň {num_placeholders} zástupných symbolov "
        "reprezentovaných hranatými zátvorkami, napríklad [adresa].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    return ["num_placeholders"]

  def check_following(self, value):
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets=None):
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet list items.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Tvoja odpoveď musí obsahovať presne {num_bullets} bodov zoznamu. "
        "Použi body zoznamu v Markdowne, napríklad:\n"
        "* Toto je bod 1. \n"
        "* Toto je bod 2")
    return self._description_pattern.format(num_bullets=self._num_bullets)

  def get_instruction_args(self):
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    return ["num_bullets"]

  def check_following(self, value):
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists)
    return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
  """Checks the constrained response."""

  def build_description(self):
    """Build the instruction description."""
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = (
        "Odpovedz jednou z nasledujúcich možností: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = value.strip()
    for constrained_response in self._constrained_responses:
      if constrained_response in value:
        return True
    return False


class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights=None, relation=None):
    """Build the instruction description.

    Args:
      num_highlights: An integer specifying the minimum number of highlighted
        sections.
      relation: A string in _COMPARISON_RELATION.

    Returns:
      A string representing the instruction description.
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    relation = _normalize_relation(relation)
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Zvýrazni {relation} {num_highlights} sekcií vo svojej odpovedi "
        "pomocou markdownu, napr. *zvýraznená sekcia*.")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_highlights=self._num_highlights)

  def get_instruction_args(self):
    return {"num_highlights": self._num_highlights,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["num_highlights", "relation"]

  def check_following(self, value):
    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", value)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
    for highlight in double_highlights:
      if highlight.removeprefix("**").removesuffix("**").strip():
        num_highlights += 1

    return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
  """Checks the sections."""

  def build_description(self, *, section_spliter=None,
                        num_sections=None, relation=None):
    """Build the instruction description.

    Args:
      section_spliter: A string representing the section splitter keyword.
      num_sections: An integer specifying the number of sections.
      relation: A string in _COMPARISON_RELATION.

    Returns:
      A string representing the instruction description.
    """
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    relation = _normalize_relation(relation)
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Tvoja odpoveď musí mať {relation} {num_sections} sekcií. "
        "Označuj začiatok každej sekcie slovom {section_spliter} X, napríklad:\n"
        "{section_spliter} 1\n"
        "[obsah sekcie 1]\n"
        "{section_spliter} 2\n"
        "[obsah sekcie 2]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter,
        relation=self._comparison_relation)

  def get_instruction_args(self):
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["section_spliter", "num_sections", "relation"]

  def check_following(self, value):
    section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
  """Checks the paragraphs."""

  def build_description(self, *, num_paragraphs=None):
    """Build the instruction description.

    Args:
      num_paragraphs: An integer specifying the number of paragraphs.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "Musí byť {num_paragraphs} odsekov. "
        "Odseky sú oddelené oddeľovačom markdownu: ***")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    return ["num_paragraphs"]

  def check_following(self, value):
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False

    return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
  """Checks the postscript."""

  def build_description(self, *, postscript_marker=None):
    """Build the instruction description.

    Args:
      postscript_marker: A string containing the keyword that marks the start
        of the postscript section.

    Returns:
      A string representing the instruction description.
    """
    self._postscript_marker = postscript_marker.strip() if isinstance(
        postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = (
        "Na konci odpovede, prosím, explicitne pridaj postskriptum "
        "začínajúce slovom {postscript}")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    return ["postscript_marker"]

  def check_following(self, value):
    value = value.lower()
    if self._postscript_marker == "P.S.":
      postscript_pattern = r"\s*p\.\s?s\..*$"
    elif self._postscript_marker == "P.P.S":
      postscript_pattern = r"\s*p\.\s?p\.\s?s[.]?.*$"
    else:
      postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
    postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
    return True if postscript else False


class KeywordChecker(Instruction):
  """Check the existence of certain keywords."""

  def build_description(self, *, keywords=None):
    """Build the instruction description.

    Args:
      keywords: A sequence of strings representing the keywords that are
        expected in the response.

    Returns:
      A string representing the instruction description.
    """
    if not keywords:
      self._keywords = sk_instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = (
        "Zahrň kľúčové slová {keywords} do svojej odpovede.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    return ["keywords"]

  def check_following(self, value):
    for keyword in self._keywords:
      if not re.search(keyword, value, flags=re.IGNORECASE):
        return False
    return True


class KeywordFrequencyChecker(Instruction):
  """Check the keyword frequency."""

  def build_description(self, *, keyword=None, frequency=None, relation=None):
    """Build the instruction description.

    Args:
      keyword: A string representing a keyword expected in the response.
      frequency: An integer specifying the number of times the keyword appears.
      relation: A string in _COMPARISON_RELATION.

    Returns:
      A string representing the instruction description.
    """
    if not keyword:
      self._keyword = sk_instructions_util.generate_keywords(
          num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    relation = _normalize_relation(relation)
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "V odpovedi sa musí slovo {keyword} objaviť {relation} "
        "{frequency} krát.")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=self._comparison_relation,
        frequency=self._frequency)

  def get_instruction_args(self):
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    actual_occurrences = len(re.findall(
        self._keyword, value, flags=re.IGNORECASE))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return actual_occurrences >= self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return actual_occurrences <= self._frequency


class NumberOfWords(Instruction):
  """Checks the number of words."""

  def build_description(self, *, num_words=None, relation=None):
    """Build the instruction description.

    Args:
      num_words: An integer specifying the number of words in the response.
      relation: A string in _COMPARISON_RELATION (or "minimálne" as alias).

    Returns:
      A string representing the instruction description.
    """
    self._num_words = num_words
    if self._num_words is None or self._num_words < 0:
      self._num_words = random.randint(
          _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT)

    relation = _normalize_relation(relation)
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Odpovedz {relation} {num_words} slovami.")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_words=self._num_words)

  def get_instruction_args(self):
    return {"num_words": self._num_words,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["num_words", "relation"]

  def check_following(self, value):
    cleaned_text = re.sub(r'(^\s*[\d]+\.\s*)|(^\s*[-*]\s*)', '', value,
                          flags=re.MULTILINE)
    cleaned_text_without_newlines = cleaned_text.replace('\n', ' ')
    num_words = sk_instructions_util.count_words(cleaned_text_without_newlines)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_words >= self._num_words
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_words <= self._num_words


class JsonFormat(Instruction):
  """Check the JSON format."""

  def build_description(self):
    self._description_pattern = (
        "Celý výstup musí byť vo formáte JSON."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError:
      return False
    return True


class ParagraphFirstWordCheck(Instruction):
  """Check the paragraph and the first word of the nth paragraph."""

  def build_description(self, num_paragraphs=None, nth_paragraph=None,
                        first_word=None):
    """Build the instruction description.

    Args:
      num_paragraphs: An integer indicating the number of paragraphs expected.
      nth_paragraph: An integer indicating the paragraph number (1-based).
      first_word: A string representing the first word of the nth paragraph.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = sk_instructions_util.generate_keywords(
          num_keywords=1)[0]
    self._first_word = self._first_word.lower()

    self._description_pattern = (
        "Musí byť {num_paragraphs} odsekov. "
        "Odseky a len odseky sú od seba oddelené dvoma "
        "zlomami riadkov. Odsek {nth_paragraph} musí začínať slovom "
        "{first_word}.")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False
    else:
      return False

    def remove_punctuation(word):
      return ''.join(char for char in word if char not in string.punctuation)

    expected_words = self._first_word.split()
    num_expected_words = len(expected_words)

    paragraph_words = [remove_punctuation(word).lower()
                       for word in paragraph.split()]

    if len(paragraph_words) < num_expected_words:
      return False

    extracted_words = paragraph_words[:num_expected_words]

    return (
        num_paragraphs == self._num_paragraphs
        and extracted_words == expected_words
    )


class ForbiddenWords(Instruction):
  """Checks that specified words are not used in response."""

  def build_description(self, forbidden_words=None):
    """Build the instruction description.

    Args:
      forbidden_words: A sequence of strings representing words that are not
        allowed in the response.

    Returns:
      A string representing the instruction description.
    """
    if not forbidden_words:
      self._forbidden_words = sk_instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "Nepoužívaj kľúčové slová {forbidden_words} vo svojej odpovedi."
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words)

  def get_instruction_args(self):
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    return ["forbidden_words"]

  def check_following(self, value):
    for word in self._forbidden_words:
      if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
        return False
    return True


class TwoResponsesChecker(Instruction):
  """Check that two responses were given."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Daj dve rôzne odpovede. Odpovede a len odpovede musia byť "
        "oddelené šiestimi hviezdičkami: ******."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    valid_responses = list()
    responses = value.split("******")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False
      else:
        valid_responses.append(response)
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    )


class RepeatPromptThenAnswer(Instruction):
  """Checks that prompt is first repeated then answered."""

  def build_description(self, *, prompt_to_repeat=None):
    """Build the instruction description.

    Args:
      prompt_to_repeat: The prompt that is meant to be repeated.

    Returns:
      A string representing the instruction description.
    """
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeat must be set.")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "Najprv zopakuj požiadavku bez zmeny, potom daj svoju odpoveď "
        "(nič nehovor pred zopakovaním požiadavky; požiadavka, ktorú je "
        "potrebné zopakovať, nezahŕňa túto vetu)."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    return ["prompt_to_repeat"]

  def check_following(self, value):
    if value.strip().lower().startswith(
        self._prompt_to_repeat.strip().lower()):
      return True
    return False


class EndChecker(Instruction):
  """Checks that the response ends with a given phrase."""

  def build_description(self, *, end_phrase=None):
    """Build the instruction description.

    Args:
      end_phrase: A string representing the phrase the response should end with.

    Returns:
      A string representing the instruction description.
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase)
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Ukonči svoju odpoveď touto presnou frázou {ender}. "
        "Za touto frázou nesmú nasledovať žiadne ďalšie slová.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    return ["end_phrase"]

  def check_following(self, value):
    value = value.strip().strip('"').lower()
    self._end_phrase = self._end_phrase.strip().lower()
    if value and value[-1] == ".":
      value = value[:-1]
    return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Tvoja odpoveď musí obsahovať nadpis obalený v dvojitých uhlových "
        "zátvorkách, napríklad <<báseň o radosti>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True
    return False


class LetterFrequencyChecker(Instruction):
  """Checks letter frequency."""

  def build_description(self, *, letter=None, let_frequency=None,
                        let_relation=None):
    """Build the instruction description.

    Args:
      letter: A string representing a letter that is expected in the response.
      let_frequency: An integer specifying the number of times the letter
        should appear.
      let_relation: A string in _COMPARISON_RELATION.

    Returns:
      A string representing the instruction description.
    """
    if (
        not letter
        or len(letter) > 1
        or ord(letter.lower()) < 97
        or ord(letter.lower()) > 122
    ):
      self._letter = random.choice(list(string.ascii_letters))
    else:
      self._letter = letter.strip()
    self._letter = self._letter.lower()

    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    let_relation = _normalize_relation(let_relation)
    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {let_relation} is given.")
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "Vo svojej odpovedi by sa písmeno {letter} malo objaviť "
        "{let_relation} {let_frequency} krát.")

    return self._description_pattern.format(
        letter=self._letter,
        let_frequency=self._frequency,
        let_relation=self._comparison_relation)

  def get_instruction_args(self):
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    value = value.lower()
    letters = collections.Counter(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return letters[self._letter] >= self._frequency
    else:
      return letters[self._letter] <= self._frequency


class CapitalLettersEnglishChecker(Instruction):
  """Checks that the response is in English and in all capital letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Celá tvoja odpoveď musí byť v angličtine, iba veľké písmená."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return value.isupper() and langdetect.detect(value) == "en"
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e)
      return True


class LowercaseLettersEnglishChecker(Instruction):
  """Checks that the response is in English and in all lowercase letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Celá tvoja odpoveď musí byť v angličtine, iba malé písmená. "
        "Veľké písmená nie sú povolené."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    assert isinstance(value, str)
    try:
      return value.islower() and langdetect.detect(value) == "en"
    except langdetect.LangDetectException as e:
      logging.error(
          "Unable to detect language for text %s due to %s", value, e)
      return True


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Vo celej odpovedi sa zdržiavaj používania čiarok."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
  """Checks frequency of words with all capital letters."""

  def build_description(self, capital_frequency=None, capital_relation=None):
    """Build the instruction description.

    Args:
      capital_frequency: An integer representing the number of fully
        capitalised words.
      capital_relation: A string in _COMPARISON_RELATION.

    Returns:
      A string representing the instruction description.
    """
    self._frequency = capital_frequency
    if self._frequency is None:
      self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

    capital_relation = _normalize_relation(capital_relation)
    self._comparison_relation = capital_relation
    if capital_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif capital_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {capital_relation} is given.")

    self._description_pattern = (
        "Vo svojej odpovedi by sa slová napísané úplne veľkými písmenami "
        "mali objaviť {relation} {frequency} krát.")

    return self._description_pattern.format(
        frequency=self._frequency, relation=self._comparison_relation)

  def get_instruction_args(self):
    return {
        "capital_frequency": self._frequency,
        "capital_relation": self._comparison_relation,
    }

  def get_instruction_args_keys(self):
    return ["capital_frequency", "capital_relation"]

  def check_following(self, value):
    words = sk_instructions_util.tokenize_words(value)
    capital_words = [word for word in words if word.isupper()]
    capital_words = len(capital_words)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return capital_words >= self._frequency
    else:
      return capital_words <= self._frequency


class QuotationChecker(Instruction):
  """Checks response is wrapped with double quotation marks."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Obal celú svoju odpoveď do úvodzoviek."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    return []

  def check_following(self, value):
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'
