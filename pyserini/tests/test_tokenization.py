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

import unittest

from transformers import BertTokenizer, T5Tokenizer, AutoTokenizer
from pyserini.analysis import Analyzer, get_lucene_analyzer


class TestTokenization(unittest.TestCase):
    def setUp(self):
        pass

    def test_bert_base_uncased_demo(self):
        # https://huggingface.co/transformers/tokenizer_summary.html
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize('I have a new GPU!')
        self.assertEqual(['i', 'have', 'a', 'new', 'gp', '##u', '!'], tokens)

    def test_bert_base_uncased_en_book_examples(self):
        # These are examples used in the ptr4tr book
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tokens = tokenizer.tokenize('walking talking balking biking hiking rolling scrolling')
        self.assertEqual(['walking', 'talking', 'bal', '##king', 'biking', 'hiking', 'rolling', 'scrolling'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['bio', '##sta', '##tist', '##ics'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['ad', '##vers', '##aria', '##l'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        tokens = tokenizer.tokenize('walking talking balking biking hiking')
        self.assertEqual(['walking', 'talking', 'b', '##alk', '##ing', 'bi', '##king', 'hiking'], tokens)

        tokens = tokenizer.tokenize('rolling scrolling')
        self.assertEqual(['rolling', 'scroll', '##ing'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['bio', '##sta', '##tist', '##ics'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['ad', '##vers', '##aria', '##l'], tokens)

    def test_xlm_roberta_base_en_book_examples(self):
        # These are examples used in the ptr4tr book
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('walking talking balking biking hiking rolling scrolling')
        self.assertEqual(['???walking', '???talking', '???bal', 'king', '???bi', 'king', '???hi', 'king', '???roll', 'ing', '???scroll', 'ing'], tokens)

        tokens = tokenizer.tokenize('rolling scrolling')
        self.assertEqual(['???roll', 'ing', '???scroll', 'ing'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['???bio', 'stat', 'istic', 's'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['???adversari', 'al'], tokens)

    def test_bert_base_multilingual_en_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('walking talking balking biking hiking rolling scrolling')
        self.assertEqual(['walking', 'talking', 'bal', '##king', 'bi', '##king', 'hi', '##king', 'rolling', 'sc', '##roll', '##ing'], tokens)

        tokens = tokenizer.tokenize('rolling scrolling')
        self.assertEqual(['rolling', 'sc', '##roll', '##ing'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['bio', '##stat', '##istic', '##s'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['ad', '##versari', '##al'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        tokens = tokenizer.tokenize('walking talking balking biking hiking')
        self.assertEqual(['walking', 'talking', 'bal', '##king', 'bi', '##king', 'hi', '##king'], tokens)

        tokens = tokenizer.tokenize('rolling scrolling')
        self.assertEqual(['rolling', 's', '##cro', '##lling'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['bio', '##stati', '##stic', '##s'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['ad', '##versari', '##al'], tokens)

    def test_lucene_analyzer_en_book_examples(self):
        analyzer = Analyzer(get_lucene_analyzer())

        tokens = analyzer.analyze('walking talking balking biking hiking rolling scrolling')
        self.assertEqual(['walk', 'talk', 'balk', 'bike', 'hike', 'roll', 'scroll'], tokens)

        tokens = analyzer.analyze('rolling scrolling')
        self.assertEqual(['roll', 'scroll'], tokens)

        tokens = analyzer.analyze('biostatistics')
        self.assertEqual(['biostatist'], tokens)

        tokens = analyzer.analyze('adversarial')
        self.assertEqual(['adversari'], tokens)

    def test_bert_base_multilingual_fr_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('marche parler v??lo randonn??e rouler d??filement')
        self.assertEqual(['marche', 'parler', 'velo', 'rand', '##onne', '##e', 'ro', '##uler', 'def', '##ile', '##ment'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('d??filement roulant')
        self.assertEqual(['def', '##ile', '##ment', 'ro', '##ulant'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('biostatistique')
        self.assertEqual(['bio', '##stat', '##istique'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('antagoniste')
        self.assertEqual(['ant', '##ago', '##niste'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('marche parler v??lo randonn??e rouler d??filement')
        self.assertEqual(['marche', 'parler', 'v', '##??l', '##o', 'rand', '##onn??e', 'ro', '##uler', 'd??', '##file', '##ment'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('d??filement roulant')
        self.assertEqual(['d??', '##file', '##ment', 'ro', '##ulant'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('biostatistique')
        self.assertEqual(['bio', '##stati', '##stique'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('antagoniste')
        self.assertEqual(['ant', '##agon', '##iste'], tokens)

    def test_lucene_analyzer_fr_book_examples(self):
        analyzer = Analyzer(get_lucene_analyzer(language='fr'))

        tokens = analyzer.analyze('marche parler v??lo randonn??e rouler d??filement')
        self.assertEqual(['march', 'parl', 'v??lo', 'randon', 'roul', 'defil'], tokens)

        tokens = analyzer.analyze('d??filement roulant')
        self.assertEqual(['defil', 'roulant'], tokens)

        tokens = analyzer.analyze('biostatistique')
        self.assertEqual(['biostatist'], tokens)

        tokens = analyzer.analyze('antagoniste')
        self.assertEqual(['antagonist'], tokens)

    def test_bert_base_multilingual_zh_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('??????????????????????????????????????????')
        self.assertEqual(['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('????????????')
        self.assertEqual(['???', '???', '???', '???'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('???????????????')
        self.assertEqual(['???', '???', '???', '???', '???'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('?????????')
        self.assertEqual(['???', '???', '???'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('??????????????????????????????????????????')
        self.assertEqual(['???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('????????????')
        self.assertEqual(['???', '???', '???', '???'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('???????????????')
        self.assertEqual(['???', '???', '???', '???', '???'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('?????????')
        self.assertEqual(['???', '???', '???'], tokens)

    def test_lucene_analyzer_zh_book_examples(self):
        analyzer = Analyzer(get_lucene_analyzer(language='zh'))

        tokens = analyzer.analyze('??????????????????????????????????????????')
        self.assertEqual(['??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????', '??????'], tokens)

        tokens = analyzer.analyze('????????????')
        self.assertEqual(['??????', '??????', '??????'], tokens)

        tokens = analyzer.analyze('???????????????')
        self.assertEqual(['??????', '??????', '??????', '??????'], tokens)

        tokens = analyzer.analyze('?????????')
        self.assertEqual(['??????', '??????'], tokens)

    def test_bert_base_multilingual_ar_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('?????????? ???????????? ???????? ???????????????? ?????????? ?????????????? ?????????? ???????????????? ??????????????')
        self.assertEqual(['????', '##??', '##????', '????????????', '??', '##????', '##??', '????', '##????', '##????', '##????', '????', '##??', '##????', '????', '##????', '##??????', '??????????', '????', '##????', '##????', '##????', '????', '##????', '##??????'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('???????????????? ??????????????')
        self.assertEqual(['????', '##????', '##????', '##????', '????', '##????', '##??????'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('?????????????? ????????????')
        self.assertEqual(['??????????????', '????', '##????', '##????'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('??????????')
        self.assertEqual(['??', '##????', '##????'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('?????????? ???????????? ???????? ???????????????? ?????????? ?????????????? ?????????? ???????????????? ??????????????')
        self.assertEqual(['????', '##??', '##????', '????????????', '??', '##????', '##??', '????', '##????', '##????????', '????', '##??', '##????', '????', '##????', '##??????', '??????????', '????', '##????', '##????', '##????', '????', '##????', '##??????'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('???????????????? ??????????????')
        self.assertEqual(['????', '##????', '##????', '##????', '????', '##????', '##??????'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('?????????????? ????????????')
        self.assertEqual(['??????????????', '????', '##????', '##????'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('??????????')
        self.assertEqual(['??', '##????', '##????'], tokens)

    def test_bert_base_multilingual_hi_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('???????????? ?????? ????????? ???????????? ????????? ???????????? ??????????????? ???????????? ???????????? ?????????????????? ?????????????????????')
        self.assertEqual(['??????', '##???', '??????', '?????????', '?????????', '??????', '???', '##???', '##???', '##???', '??????', '##?????????', '???', '##??????', '??????', '##???', '???????????????', '??????', '##???', '##??????'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('?????????????????? ?????????????????????')
        self.assertEqual(['???', '##??????', '##??????', '??????', '##???', '##??????'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('????????? ???????????????????????????')
        self.assertEqual(['???', '##???', '???', '##???', '##???', '##?????????', '##???'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('??????????????????????????????')
        self.assertEqual(['??????', '##??????', '##??????', '##?????????'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('???????????? ?????? ????????? ???????????? ????????? ???????????? ??????????????? ???????????? ???????????? ?????????????????? ?????????????????????')
        self.assertEqual(['???', '##??????', '##???', '??????', '?????????', '????????????', '?????????', '???', '##???', '##???', '##???', '???', '##??????', '##??????', '???', '##???', '##??????', '???', '##???', '##??????', '??????????????????', '???', '##??????', '##??????', '##??????'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('?????????????????? ?????????????????????')
        self.assertEqual(['???', '##??????', '##?????????', '???', '##??????', '##??????', '##??????'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('????????? ???????????????????????????')
        self.assertEqual(['???', '##???', '##???', '???', '##???', '##???', '##???', '##??????', '##?????????'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('??????????????????????????????')
        self.assertEqual(['??????', '##??????', '##??????', '##????????????'], tokens)

    def test_bert_base_multilingual_bn_book_examples(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('???????????????????????? ?????????????????? ?????????????????? ??????????????? ???????????????????????????')
        self.assertEqual(['???', '##??????', '##???', '##???', '##??????', '??????', '##???', '##??????', '##???', '???', '##??????', '##??????', '##???', '???', '##??????', '##??????', '##???', '???', '##??????', '##??????', '##??????', '##???'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('????????????????????????????????? ???????????????????????????')
        self.assertEqual(['??????', '##??????', '##???', '##?????????', '???', '##??????', '##??????', '##??????', '##???'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('??????????????????????????????????????????')
        self.assertEqual(['??????', '##??????', '##??????', '##??????', '##??????', '##???', '##???'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('????????????????????????')
        self.assertEqual(['????????????', '##???', '##???'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # walking talking biking hiking rolling scrolling
        tokens = tokenizer.tokenize('???????????????????????? ?????????????????? ?????????????????? ??????????????? ???????????????????????????')
        self.assertEqual(['???', '##???', '##???', '##??????', '##???', '##??????', '??????', '##???', '##??????', '##???', '???', '##??????', '##??????', '##???', '???', '##??????', '##??????', '???', '##??????', '##??????', '##??????', '##??????'], tokens)

        # rolling scrolling
        tokens = tokenizer.tokenize('????????????????????????????????? ???????????????????????????')
        self.assertEqual(['???', '##????????????', '##?????????', '##?????????', '???', '##??????', '##??????', '##??????', '##??????'], tokens)

        # biostatistics
        tokens = tokenizer.tokenize('??????????????????????????????????????????')
        self.assertEqual(['??????', '##??????', '##???', '##?????????', '##??????', '##??????', '##??????'], tokens)

        # adversarial
        tokens = tokenizer.tokenize('????????????????????????')
        self.assertEqual(['???????????????', '##???', '##???', '##???'], tokens)
    
    def test_bert_base_multilingual_am(self):
        """
        amharic
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('???????????? ????????? ????????? ????????? ????????? ????????????')
        self.assertEqual(['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'], tokens)

        tokens = tokenizer.tokenize('????????????')
        self.assertEqual(['[UNK]'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('???????????? ????????? ????????? ????????? ????????? ????????????')
        self.assertEqual(['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'], tokens)

        tokens = tokenizer.tokenize('????????????')
        self.assertEqual(['[UNK]'], tokens)
    
    def test_xlmr_base_multilingual_am(self):
        """
        amharic
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('???????????? ????????? ????????? ????????? ????????? ????????????')
        self.assertEqual(['??????', '???', '???', '???', '????????????', '????????????', '??????', '??????', '????????????', '??????', '??????', '???'], tokens)

        tokens = tokenizer.tokenize('????????????')
        self.assertEqual(['??????', '???', '???', '???'], tokens)
    
    def test_bert_base_multilingual_ha(self):
        """
        hausa
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Ya san kungiyar, ya san komai game da kungiyar')
        self.assertEqual(['ya', 'san', 'kung', '##iya', '##r', ',', 'ya', 'san', 'koma', '##i', 'game', 'da', 'kung', '##iya', '##r'], tokens)

        tokens = tokenizer.tokenize('kungiyar')
        self.assertEqual(['kung', '##iya', '##r'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Ya san kungiyar, ya san komai game da kungiyar')
        self.assertEqual(['Ya', 'san', 'kung', '##iya', '##r', ',', 'ya', 'san', 'koma', '##i', 'game', 'da', 'kung', '##iya', '##r'], tokens)

        tokens = tokenizer.tokenize('kungiyar')
        self.assertEqual(['kung', '##iya', '##r'], tokens)
    
    def test_xlmr_base_multilingual_ha(self):
        """
        hausa
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Ya san kungiyar, ya san komai game da kungiyar')
        self.assertEqual(['???Ya', '???san', '???kungiyar', ',', '???ya', '???san', '???koma', 'i', '???game', '???da', '???kungiyar'], tokens)

        tokens = tokenizer.tokenize('kungiyar')
        self.assertEqual(['???kungiyar'], tokens)

    def test_bert_base_multilingual_ig(self):
        """
        igbo
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Oke ???????? Adaa D???ka L??????l??? Ezenneka gb??r?? Ah??? Otu Nar???')
        self.assertEqual(['ok', '##e', 'onu', 'ada', '##a', 'dik', '##a', 'lo', '##olo', 'ezen', '##nek', '##a', 'gba', '##ra', 'ah', '##o', 'ot', '##u', 'nar', '##i'], tokens)

        tokens = tokenizer.tokenize('Ezenneka')
        self.assertEqual(['ezen', '##nek', '##a'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Oke ???????? Adaa D???ka L??????l??? Ezenneka gb??r?? Ah??? Otu Nar???')
        self.assertEqual(['Ok', '##e', '???', '##??', '##???', 'Ada', '##a', 'D', '##???', '##ka', 'L', '##???', '##???', '##l', '##???', 'Ezen', '##nek', '##a', 'g', '##b??', '##r??', 'Ah', '##???', 'O', '##tu', 'Na', '##r', '##???'], tokens)

        tokens = tokenizer.tokenize('Ezenneka')
        self.assertEqual(['Ezen', '##nek', '##a'], tokens)
    
    def test_xlmr_base_multilingual_ig(self):
        """
        igbo
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Oke ???????? Adaa D???ka L??????l??? Ezenneka gb??r?? Ah??? Otu Nar???')
        self.assertEqual(['???O', 'ke', '???', '???', '??', '???', '???Ada', 'a', '???D', '???', 'ka', '???L', '???', '???', 'l', '???', '???Ezen', 'nek', 'a', '???', 'gb', '??', 'r??', '???Ah', '???', '???O', 'tu', '???Nar', '???'], tokens)

        tokens = tokenizer.tokenize('Ezenneka')
        self.assertEqual(['???Ezen', 'nek', 'a'], tokens)

    def test_bert_base_multilingual_om(self):
        """
        Afaan Oromoo
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Ani obbolaa keessan, Abdii Baalee Oromiyaatii')
        self.assertEqual(['ani', 'ob', '##bola', '##a', 'ke', '##essa', '##n', ',', 'abd', '##ii', 'ba', '##ale', '##e', 'oro', '##mi', '##ya', '##atii'], tokens)

        tokens = tokenizer.tokenize('Oromiyaatii')
        self.assertEqual(['oro', '##mi', '##ya', '##atii'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Ani obbolaa keessan, Abdii Baalee Oromiyaatii')
        self.assertEqual(['Ani', 'ob', '##bola', '##a', 'ke', '##essa', '##n', ',', 'Abd', '##ii', 'Ba', '##ale', '##e', 'Oro', '##mi', '##ya', '##ati', '##i'], tokens)

        tokens = tokenizer.tokenize('Oromiyaatii')
        self.assertEqual(['Oro', '##mi', '##ya', '##ati', '##i'], tokens)
    
    def test_xlmr_base_multilingual_om(self):
        """
        Afaan Oromoo
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Ani obbolaa keessan, Abdii Baalee Oromiyaatii')
        self.assertEqual(['???Ani', '???ob', 'bola', 'a', '???keessa', 'n', ',', '???Ab', 'dii', '???Ba', 'ale', 'e', '???Oromiyaa', 'tii'], tokens)

        tokens = tokenizer.tokenize('Oromiyaatii')
        self.assertEqual(['???Oromiyaa', 'tii'], tokens)

    def test_bert_base_multilingual_pcm(self):
        """
        Nigerian Pidgin
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Crude oil dey kill pickin for Nigeria?')
        self.assertEqual(['cru', '##de', 'oil', 'de', '##y', 'kill', 'pick', '##in', 'for', 'nigeria', '?'], tokens)

        tokens = tokenizer.tokenize('wahala')
        self.assertEqual(['wah', '##ala'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Crude oil dey kill pickin for Nigeria?')
        self.assertEqual(['C', '##rude', 'oil', 'de', '##y', 'kill', 'pick', '##in', 'for', 'Nigeria', '?'], tokens)

        tokens = tokenizer.tokenize('wahala')
        self.assertEqual(['wa', '##hala'], tokens)
    
    def test_xlmr_base_multilingual_pcm(self):
        """
        Nigerian Pidgin
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Crude oil dey kill pickin for Nigeria?')
        self.assertEqual(['???Cru', 'de', '???oil', '???de', 'y', '???kill', '???pick', 'in', '???for', '???Nigeria', '?'], tokens)

        tokens = tokenizer.tokenize('wahala')
        self.assertEqual(['???wa', 'hala'], tokens)

    def test_bert_base_multilingual_so(self):
        """
        Somali
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Rabbigu wuxuu amar ku bixiyey in la dumiyo qalcadaha Kancaan.')
        self.assertEqual(['rabbi', '##gu', 'wu', '##xu', '##u', 'amar', 'ku', 'bi', '##xi', '##ye', '##y', 'in', 'la', 'dum', '##iy', '##o', 'qal', '##cada', '##ha', 'kan', '##ca', '##an', '.'], tokens)

        tokens = tokenizer.tokenize('bixiyey')
        self.assertEqual(['bi', '##xi', '##ye', '##y'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Rabbigu wuxuu amar ku bixiyey in la dumiyo qalcadaha Kancaan.')
        self.assertEqual(['Rabbi', '##gu', 'w', '##ux', '##uu', 'amar', 'ku', 'bi', '##xi', '##ye', '##y', 'in', 'la', 'dum', '##iyo', 'q', '##al', '##cada', '##ha', 'Kan', '##ca', '##an', '.'], tokens)

        tokens = tokenizer.tokenize('bixiyey')
        self.assertEqual(['bi', '##xi', '##ye', '##y'], tokens)
    
    def test_xlmr_base_multilingual_so(self):
        """
        Somali
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Rabbigu wuxuu amar ku bixiyey in la dumiyo qalcadaha Kancaan.')
        self.assertEqual(['???Rabbi', 'gu', '???wuxuu', '???amar', '???ku', '???bixi', 'yey', '???in', '???la', '???dum', 'iyo', '???qal', 'cada', 'ha', '???Kan', 'ca', 'an', '.'], tokens)

        tokens = tokenizer.tokenize('bixiyey')
        self.assertEqual(['???bixi', 'yey'], tokens)

    def test_bert_base_multilingual_sw(self):
        """
        Swahili
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Huduma ya upasuaji mkubwa na mdogo')
        self.assertEqual(['hu', '##dum', '##a', 'ya', 'up', '##asu', '##aji', 'mk', '##ubwa', 'na', 'md', '##ogo'], tokens)

        tokens = tokenizer.tokenize('upasuaji')
        self.assertEqual(['up', '##asu', '##aji'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Huduma ya upasuaji mkubwa na mdogo')
        self.assertEqual(['Hu', '##dum', '##a', 'ya', 'up', '##asu', '##aji', 'mk', '##ub', '##wa', 'na', 'm', '##dogo'], tokens)

        tokens = tokenizer.tokenize('upasuaji')
        self.assertEqual(['up', '##asu', '##aji'], tokens)
    
    def test_xlmr_base_multilingual_sw(self):
        """
        Swahili
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Huduma ya upasuaji mkubwa na mdogo')
        self.assertEqual(['???Huduma', '???ya', '???up', 'asu', 'aji', '???mkubwa', '???na', '???mdogo'], tokens)

        tokens = tokenizer.tokenize('upasuaji')
        self.assertEqual(['???up', 'asu', 'aji'], tokens)

    def test_bert_base_multilingual_ti(self):
        """
        Tigrinya
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('???????????? ????????? ??????????????? ???????????? ????????? ????????? ????????????')
        self.assertEqual(['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'], tokens)

        tokens = tokenizer.tokenize('???????????????')
        self.assertEqual(['[UNK]'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('???????????? ????????? ??????????????? ???????????? ????????? ????????? ????????????')
        self.assertEqual(['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]'], tokens)

        tokens = tokenizer.tokenize('???????????????')
        self.assertEqual(['[UNK]'], tokens)
    
    def test_xlmr_base_multilingual_ti(self):
        """
        Tigrinya
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('???????????? ????????? ??????????????? ???????????? ????????? ????????? ????????????')
        self.assertEqual(['?????????', '???', '???', '??????', '??????', '?????????', '???', '???', '???', '??????', '???', '???', '???', '????????????', '??????', '???', '???', '?????????', '???', '???'], tokens)

        tokens = tokenizer.tokenize('???????????????')
        self.assertEqual(['?????????', '???', '???', '???'], tokens)

    def test_bert_base_multilingual_yo(self):
        """
        Yoruba
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokens = tokenizer.tokenize('Or??k??? ???m???binrin r????? ??gb?? ni Merabu, ti ??y?? ??b??r?? ni Mikali.')
        self.assertEqual(['oru', '##ko', 'omo', '##bin', '##rin', 're', 'ag', '##ba', 'ni', 'mera', '##bu', ',', 'ti', 'e', '##yi', 'abu', '##ro', 'ni', 'mika', '##li', '.'], tokens)

        tokens = tokenizer.tokenize('???m???binrin')
        self.assertEqual(['omo', '##bin', '##rin'], tokens)

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        tokens = tokenizer.tokenize('Or??k??? ???m???binrin r????? ??gb?? ni Merabu, ti ??y?? ??b??r?? ni Mikali.')
        self.assertEqual(['Or', '##??', '##k', '##???', '???', '##m', '##???', '##bin', '##rin', 'r', '##?????', '??', '##g', '##b??', 'ni', 'Mer', '##abu', ',', 'ti', '??', '##y', '##??', '??', '##b', '##??r', '##??', 'ni', 'Mika', '##li', '.'], tokens)

        tokens = tokenizer.tokenize('???m???binrin')
        self.assertEqual(['???', '##m', '##???', '##bin', '##rin'], tokens)
    
    def test_xlmr_base_multilingual_yo(self):
        """
        Yoruba
        """
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

        tokens = tokenizer.tokenize('Or??k??? ???m???binrin r????? ??gb?? ni Merabu, ti ??y?? ??b??r?? ni Mikali.')
        self.assertEqual(['???O', 'r??', 'k', '???', '???', '???', 'm', '???', 'bin', 'rin', '???r', '???', '??', '?????', 'gb', '??', '???ni', '???Mera', 'bu', ',', '???ti', '?????', 'y', '??', '?????', 'b??', 'r??', '???ni', '???Mi', 'kali', '.'], tokens)

        tokens = tokenizer.tokenize('???m???binrin')
        self.assertEqual(['???', '???', 'm', '???', 'bin', 'rin'], tokens)

    def test_doc2query(self):
        tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        tokens = tokenizer.tokenize('I have a new GPU!')
        self.assertEqual(['???I', '???have', '???', 'a', '???new', '???GPU', '!'], tokens)

        tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
        tokens = tokenizer.tokenize('walking talking biking scrolling')
        self.assertEqual(['???walking', '???talking', '???biking', '???scroll', 'ing'], tokens)

        tokens = tokenizer.tokenize('biostatistics')
        self.assertEqual(['???bio', 'stat', 'istic', 's'], tokens)

        tokens = tokenizer.tokenize('adversarial')
        self.assertEqual(['???adversar', 'i', 'al'], tokens)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
