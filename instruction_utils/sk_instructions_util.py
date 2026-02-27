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

"""Utility library of Slovak instructions."""

# Uses the multilingual spaCy model (xx_sent_ud_sm) as no dedicated Slovak
# model is available. Install with:
#   pip install spacy
#   python -m spacy download xx_sent_ud_sm

import spacy

import random
import re
from typing import List

import immutabledict


WORD_LIST = ["priateľ", "jedlo", "škola", "dom", "rodina", "práca", "čas", "kniha", "mesto", "pes"]  # pylint: disable=line-too-long

# ISO 639-1 codes to language names in Slovak.
LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "Angličtina",
    "es": "Španielčina",
    "pt": "Portugalčina",
    "ar": "Arabčina",
    "hi": "Hindčina",
    "fr": "Francúzština",
    "ru": "Ruština",
    "de": "Nemčina",
    "ja": "Japončina",
    "it": "Taliančina",
    "bn": "Bengálčina",
    "uk": "Ukrajiničina",
    "th": "Thajčina",
    "ur": "Urdčina",
    "ta": "Tamilčina",
    "te": "Telugčina",
    "bg": "Bulharčina",
    "ko": "Kórejčina",
    "pl": "Poľčina",
    "he": "Hebrejčina",
    "fa": "Perziančina",
    "vi": "Vietnamčina",
    "ne": "Nepálčina",
    "sw": "Svahilčina",
    "kn": "Kannada",
    "mr": "Maráthčina",
    "gu": "Gudžarátčina",
    "pa": "Pandžábčina",
    "ml": "Malajálčina",
    "fi": "Fínčina",
    "sk": "Slovenčina",
    })

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Dr|Mgr|Ing|PhDr|RNDr|MUDr|JUDr|PaedDr|Doc|Prof|Bc|Mr|St|Mrs|Ms|atď|resp|napr|tzv|vid|pozn)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Dr|Mgr|Ing|Prof|Doc|Mr|Mrs|Ms|On\s|Ona\s|To\s|Oni\s|Ony\s|Jeho\s|Náš\s|Naša\s|My\s|Ale\s|Avšak\s|Ten\s|Tá\s|Tento\s|Táto\s|Kdekoľvek\s|Pokiaľ ide o\s|Preto\s|Napríklad\s|Stručne povedané\s|V dôsledku toho\s|Na druhej strane\s|Vzhľadom na\s)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me|sk|cz)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"

nlp = spacy.load("xx_sent_ud_sm")


def split_into_sentences(text):
  """Split the text into sentences.

  Args:
    text: A string that consists of more than or equal to one sentences.

  Returns:
    A list of strings where each string is a sentence.
  """
  text = " " + text + "  "
  text = text.replace("\n", " ")
  text = re.sub(_PREFIXES, "\\1<prd>", text)
  text = re.sub(_WEBSITES, "<prd>\\1", text)
  text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
  text = re.sub(
      _MULTIPLE_DOTS,
      lambda match: "<prd>" * len(match.group(0)) + "<stop>",
      text,
  )
  if "Ph.D" in text:
    text = text.replace("Ph.D.", "Ph<prd>D<prd>")
  text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
  text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
  text = re.sub(
      _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
      "\\1<prd>\\2<prd>\\3<prd>",
      text,
  )
  text = re.sub(
      _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
  )
  text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
  text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
  text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
  if "\u201c" in text:
    text = text.replace(".\u201d", "\u201d.")
  if '"' in text:
    text = text.replace('."', '".')
  if "!" in text:
    text = text.replace('!"', '"!')
  if "?" in text:
    text = text.replace('?"', '"?')
  text = text.replace(".", ".<stop>")
  text = text.replace("?", "?<stop>")
  text = text.replace("!", "!<stop>")
  text = text.replace("<prd>", ".")
  sentences = text.split("<stop>")
  sentences = [s.strip() for s in sentences]
  if sentences and not sentences[-1]:
    sentences = sentences[:-1]
  return sentences


def count_words(text):
  """Counts the number of words using the multilingual spaCy model."""
  tokenized_text = nlp(text)
  num_words = len([token.text for token in tokenized_text if not token.is_punct])
  return num_words


def tokenize_words(text):
  """Returns a list of non-punctuation words using the multilingual spaCy model."""
  tokenized_text = nlp(text)
  words = [token.text for token in tokenized_text if not token.is_punct]
  return words


def count_sentences(text):
  """Count the number of sentences using the multilingual spaCy model."""
  tokenized_text = nlp(text)
  num_sentences = len(list(tokenized_text.sents))
  return num_sentences


def generate_keywords(num_keywords):
  """Randomly generates a few keywords."""
  return random.sample(WORD_LIST, k=num_keywords)
