#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This module provides Pyserini's Python interface for raw access to Lucene indexes built by Anserini. The main entry
point is the ``IndexReaderUtils`` class, which wraps the Java class with the same name in Anserini. Many of the classes
and methods provided are meant only to provide tools for examining an index and are not optimized for computing over.
"""

import logging
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

from ..analysis import get_lucene_analyzer, JAnalyzer, JAnalyzerUtils
from ..pyclass import autoclass, JString
from ..search import Document
from pyserini.util import download_prebuilt_index, get_sparse_indexes_info
from pyserini.prebuilt_index_info import INDEX_INFO

logger = logging.getLogger(__name__)


# Wrappers around Anserini classes
JIndexReader = autoclass('io.anserini.index.IndexReaderUtils')


class JIndexHelpers:
    def JArgs():
        args = autoclass('io.anserini.index.IndexArgs')()
        args.storeContents = True
        args.storeRaw = True
        args.dryRun = True ## So that indexing will be skipped
        return args

    def JCounters():
        IndexCollection = autoclass('io.anserini.index.IndexCollection')
        Counters = autoclass('io.anserini.index.IndexCollection$Counters')
        return Counters(IndexCollection)


class JGenerators(Enum):
    AclAnthologyGenerator = autoclass('io.anserini.index.generator.AclAnthologyGenerator')
    DefaultLuceneDocumentGenerator = autoclass('io.anserini.index.generator.DefaultLuceneDocumentGenerator')
    TweetGenerator = autoclass('io.anserini.index.generator.TweetGenerator')
    WashingtonPostGenerator = autoclass('io.anserini.index.generator.WashingtonPostGenerator')


class Generator:
    """Wrapper class for Anserini's generators.

    Parameters
    ----------
    generator_class : str
        Name of generator class to instantiate
    """

    def __init__(self, generator_class):
        self.counters = JIndexHelpers.JCounters()
        self.args = JIndexHelpers.JArgs()
        self.generator_class = generator_class
        self.object = self._get_generator()

    def _get_generator(self):
        try:
            return JGenerators[self.generator_class].value(self.args)
        except:
            raise ValueError(self.generator_class)

    def create_document(self, document):
        """
        Parameters
        ----------
        document : pyserini.collection.pycollection.Document
            Collection document to create Lucene document from

        Returns
        -------
        result : org.apache.lucene.document.Document
            Lucene document generated
        """
        return self.object.createDocument(document.object)


class IndexTerm:
    """Class representing an analyzed term in an index with associated statistics.

    Parameters
    ----------
    term : str
        Analyzed term.
    df : int
        Document frequency, the number of documents in the collection that contains the term.
    cf : int
        Collection frequency, the number of times the term occurs in the entire collection.  This value is equal to the
        sum of all the term frequencies of the term across all documents in the collection.
    """

    def __init__(self, term, df, cf):
        self.term = term
        self.df = df
        self.cf = cf


class Posting:
    """Class representing a posting in a postings list.

    Parameters
    ----------
    docid : int
        Collection ``docid``.
    tf : int
        Term frequency.
    positions : List[int]
        List of positions.
    """

    def __init__(self, docid, tf, positions):
        self.docid = docid
        self.tf = tf
        self.positions = positions

    def __repr__(self):
        repr = '(' + str(self.docid) + ', ' + str(self.tf) + ')'
        if self.positions:
            repr += ' [' + ','.join([str(p) for p in self.positions]) + ']'
        return repr


class IndexReader:
    """Wrapper class for ``IndexReaderUtils`` in Anserini.

    Parameters
    ----------
    index_dir : str
        Path to Lucene index directory.
    """

    def __init__(self, index_dir):
        self.object = JIndexReader()
        self.reader = self.object.getReader(JString(index_dir))

    @classmethod
    def from_prebuilt_index(cls, prebuilt_index_name: str):
        """Build an index reader from a prebuilt index; download the index if necessary.

        Parameters
        ----------
        prebuilt_index_name : str
            Prebuilt index name.

        Returns
        -------
        IndexReader
            Index reader built from the prebuilt index.
        """
        print(f'Attempting to initialize pre-built index {prebuilt_index_name}.')
        try:
            index_dir = download_prebuilt_index(prebuilt_index_name)
        except ValueError as e:
            print(str(e))
            return None

        print(f'Initializing {prebuilt_index_name}...')
        return cls(index_dir)

    @classmethod
    def validate_prebuilt_index(cls, prebuilt_index_name: str):
        """Validate prebuilt index stats against stored stats."""
        reader = cls.from_prebuilt_index(prebuilt_index_name)
        stats = reader.stats()

        if stats['documents'] != INDEX_INFO[prebuilt_index_name]['documents']:
            raise ValueError('"documents" does not match!')

        if stats['unique_terms'] != INDEX_INFO[prebuilt_index_name]['unique_terms']:
            raise ValueError('"unique_terms" does not match!')

        if stats['total_terms'] != INDEX_INFO[prebuilt_index_name]['total_terms']:
            raise ValueError('"total_terms" does not match!')

        print(reader.stats())
        print('Statistics match!')

        return True

    @staticmethod
    def list_prebuilt_indexes():
        """Display information about available prebuilt indexes."""
        get_sparse_indexes_info()

    def analyze(self, text: str, analyzer=None) -> List[str]:
        """Analyze a piece of text. Applies Anserini's default Lucene analyzer if analyzer not specified.

        Parameters
        ----------
        text : str
            Text to analyze.
        analyzer : analyzer
            Analyzer to apply.
        Returns
        -------
        List[str]
            List of tokens corresponding to the output of the analyzer.
        """
        if analyzer is None:
            results = JAnalyzerUtils.analyze(JString(text.encode('utf-8')))
        else:
            results = JAnalyzerUtils.analyze(analyzer, JString(text.encode('utf-8')))
        tokens = []
        for token in results.toArray():
            tokens.append(token)
        return tokens

    def terms(self) -> Iterator[IndexTerm]:
        """Return an iterator over analyzed terms in the index.

        Returns
        -------
        Iterator[IndexTerm]
            Iterator over :class:`IndexTerm` objects corresponding to (analyzed) terms in the index.
        """
        term_iterator = self.object.getTerms(self.reader)
        while term_iterator.hasNext():
            cur_term = term_iterator.next()
            yield IndexTerm(cur_term.getTerm(), cur_term.getDF(), cur_term.getTotalTF())

    def get_term_counts(self, term: str, analyzer: Optional[JAnalyzer] = get_lucene_analyzer()) -> Tuple[int, int]:
        """Return the document frequency and collection frequency of a term. Applies Anserini's default Lucene
        ``Analyzer`` if analyzer is not specified.

        Parameters
        ----------
        term : str
            Unanalyzed term.
        analyzer : analyzer
            Analyzer to apply.

        Returns
        -------
        Tuple[int, int]
            Document frequency and collection frequency.
        """
        if analyzer is None:
            analyzer = get_lucene_analyzer(stemming=False, stopwords=False)

        term_map = self.object.getTermCountsWithAnalyzer(self.reader, JString(term.encode('utf-8')), analyzer)

        return term_map.get(JString('docFreq')), term_map.get(JString('collectionFreq'))

    def get_postings_list(self, term: str, analyzer=get_lucene_analyzer()) -> List[Posting]:
        """Return the postings list for a term.

        Parameters
        ----------
        term : str
            Raw term.
        analyzer : analyzer
            Analyzer to apply. Defaults to Anserini's default.

        Returns
        -------
        List[Posting]
            List of :class:`Posting` objects corresponding to the postings list for the term.
        """
        if analyzer is None:
            postings_list = self.object.getPostingsListForAnalyzedTerm(self.reader, JString(term.encode('utf-8')))
        else:
            postings_list = self.object.getPostingsListWithAnalyzer(self.reader, JString(term.encode('utf-8')),
                                                                    analyzer)

        if postings_list is None:
            return None

        result = []
        for posting in postings_list.toArray():
            result.append(Posting(posting.getDocid(), posting.getTF(), posting.getPositions()))
        return result

    def get_document_vector(self, docid: str) -> Optional[Dict[str, int]]:
        """Return the document vector for a ``docid``. Note that requesting the document vector of a ``docid`` that
        does not exist in the index will return ``None`` (as opposed to an empty dictionary); this forces the caller
        to handle ``None`` explicitly and guards against silent errors.

        Parameters
        ----------
        docid : str
            Collection ``docid``.

        Returns
        -------
        Optional[Dict[str, int]]
            A dictionary with analyzed terms as keys and their term frequencies as values.
        """
        doc_vector_map = self.object.getDocumentVector(self.reader, JString(docid))
        if doc_vector_map is None:
            return None
        doc_vector_dict = {}
        for term in doc_vector_map.keySet().toArray():
            doc_vector_dict[term] = doc_vector_map.get(JString(term.encode('utf-8')))
        return doc_vector_dict

    def get_term_positions(self, docid: str) -> Optional[Dict[str, int]]:
        """Return the term position mapping of the document with ``docid``. Note that the term in the document is
        stemmed and stop words may be removed according to your index settings. Also, requesting the document vector of
        a ``docid`` that does not exist in the index will return ``None`` (as opposed to an empty dictionary); this
        forces the caller to handle ``None`` explicitly and guards against silent errors.

        Parameters
        ----------
        docid : str
            Collection ``docid``.

        Returns
        -------
        Optional[Dict[str, int]]
            A tuple contains a dictionary with analyzed terms as keys and corresponding posting list as values
        """
        java_term_position_map = self.object.getTermPositions(self.reader, JString(docid))
        if java_term_position_map is None:
            return None
        term_position_map = {}
        for term in java_term_position_map.keySet().toArray():
            term_position_map[term] = java_term_position_map.get(JString(term.encode('utf-8'))).toArray()
        return term_position_map

    def doc(self, docid: str) -> Optional[Document]:
        """Return the :class:`Document` corresponding to ``docid``. Returns ``None`` if the ``docid`` does not exist
        in the index.

        Parameters
        ----------
        docid : str
            The collection ``docid``.

        Returns
        -------
        Optional[Document]
            :class:`Document` corresponding to the ``docid``.
        """
        lucene_document = self.object.document(self.reader, JString(docid))
        if lucene_document is None:
            return None
        return Document(lucene_document)

    def doc_by_field(self, field: str, q: str) -> Optional[Document]:
        """Return the :class:`Document` based on a ``field`` with ``id``. For example, this method can be used to fetch
        document based on alternative primary keys that have been indexed, such as an article's DOI.

        Parameters
        ----------
        field : str
            The field to look up.
        q : str
            The document's unique id.

        Returns
        -------
        Optional[Document]
            :class:`Document` whose ``field`` is ``id``.
        """
        lucene_document = self.object.documentByField(self.reader, JString(field), JString(q))
        if lucene_document is None:
            return None
        return Document(lucene_document)

    def doc_raw(self, docid: str) -> Optional[str]:
        """Return the raw document contents for a collection ``docid``.

        Parameters
        ----------
        docid : str
            Collection ``docid``.

        Returns
        -------
        Optional[str]
            Raw document contents.
        """
        return self.object.documentRaw(self.reader, JString(docid))

    def doc_contents(self, docid: str) -> Optional[str]:
        """Return the indexed document contents for a collection ``docid``.

        Parameters
        ----------
        docid : str
            The collection ``docid``.

        Returns
        -------
        Optional[str]
            Index document contents.
        """
        return self.object.documentContents(self.reader, JString(docid))

    def compute_bm25_term_weight(self, docid: str, term: str, analyzer=get_lucene_analyzer(), k1=0.9, b=0.4) -> float:
        """Compute the BM25 weight of a term in a document. Specify ``analyzer=None`` for an already analyzed term,
        e.g., from the output of :func:`get_document_vector`.

        Parameters
        ----------
        docid : str
            Collection ``docid``.
        term : str
            Term.
        analyzer : analyzer
            Lucene analyzer to use, ``None`` if term is already analyzed.
        k1 : float
            BM25 k1 parameter.
        b : float
            BM25 b parameter.

        Returns
        -------
        float
            BM25 weight of the term in the document, or 0 if the term does not exist in the document.
        """
        if analyzer is None:
            return self.object.getBM25AnalyzedTermWeightWithParameters(self.reader, JString(docid),
                                                                       JString(term.encode('utf-8')),
                                                                       float(k1), float(b))
        else:
            return self.object.getBM25UnanalyzedTermWeightWithParameters(self.reader, JString(docid),
                                                                         JString(term.encode('utf-8')), analyzer,
                                                                         float(k1), float(b))

    def compute_query_document_score(self, docid: str, query: str, similarity=None):
        if similarity is None:
            return self.object.computeQueryDocumentScore(self.reader, docid, query)
        else:
            return self.object.computeQueryDocumentScoreWithSimilarity(self.reader, docid, query, similarity)

    def convert_internal_docid_to_collection_docid(self, docid: int) -> str:
        """Convert Lucene's internal ``docid`` to its external collection ``docid``.

        Parameters
        ----------
        docid : int
            Lucene internal ``docid``.

        Returns
        -------
        str
            External collection ``docid`` corresponding to Lucene's internal ``docid``.
        """
        return self.object.convertLuceneDocidToDocid(self.reader, docid)

    def convert_collection_docid_to_internal_docid(self, docid: str) -> int:
        """Convert external collection ``docid`` to its Lucene's internal ``docid``.

        Parameters
        ----------
        docid : str
            External collection ``docid``.

        Returns
        -------
        str
            Lucene internal ``docid`` corresponding to the external collection ``docid``.
        """
        return self.object.convertDocidToLuceneDocid(self.reader, docid)

    def stats(self) -> Dict[str, int]:
        """Return dictionary with index statistics.

        Returns
        -------
        Dict[str, int]
            Index statistics as a dictionary of statistic's name to statistic.
            - documents: number of documents
            - non_empty_documents: number of non-empty documents
            - unique_terms: number of unique terms
            - total_terms: number of total terms
        """
        index_stats_map = self.object.getIndexStats(self.reader)

        if index_stats_map is None:
            return None

        index_stats_dict = {}
        for term in index_stats_map.keySet().toArray():
            index_stats_dict[term] = index_stats_map.get(JString(term.encode('utf-8')))

        return index_stats_dict
