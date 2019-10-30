
import numpy

from allennlp.common.testing.test_case import AllenNlpTestCase

from kb.wiki_linking_reader import LinkingReader
from kb.wiki_linking_util import WikiCandidateMentionGenerator

from kb.bert_tokenizer_and_candidate_generator import BertTokenizerAndCandidateGenerator
from kb.wordnet import WordNetCandidateMentionGenerator

from allennlp.common import Params
from allennlp.data import TokenIndexer, Vocabulary, DataIterator, Instance


class MentionGeneratorTest(AllenNlpTestCase):

    def test_read(self):

        candidate_generator = WikiCandidateMentionGenerator("tests/fixtures/linking/priors.txt")
        assert len(candidate_generator.p_e_m) == 50

        assert set(candidate_generator.p_e_m.keys()) == {
                'United States', 'Information', 'Wiki', 'France', 'English', 'Germany',
                'World War II', '2007', 'England', 'American', 'Canada', 'Australia',
                'Japan', '2008', 'India', '2006', 'Area Info', 'London', 'German',
                'About Company', 'French', 'United Kingdom', 'Italy', 'en', 'California',
                'China', '2005', 'New York', 'Spain', 'Europe', 'British', '2004',
                'New York City', 'Russia', 'public domain', '2000', 'Brazil', 'Poland',
                'micro-blogging', 'Greek', 'New Zealand', '2003', 'Mexico', 'Italian',
                'Ireland', 'Wiki Image', 'Paris', 'USA', '[1]', 'Iran'
                }


        lower = candidate_generator.process("united states")
        string_list = candidate_generator.process(["united", "states"])
        upper = candidate_generator.process(["United", "States"])
        assert lower == upper == string_list


class WikiReaderTest(AllenNlpTestCase):
    def test_wiki_linking_reader_with_wordnet(self):
        def _get_indexer(namespace):
            return TokenIndexer.from_params(Params({
                        "type": "characters_tokenizer",
                        "tokenizer": {
                                "type": "word",
                                "word_splitter": {"type": "just_spaces"},
                        },
                        "namespace": namespace
            }))

        extra_generator = {
            'wordnet': WordNetCandidateMentionGenerator(
                    'tests/fixtures/wordnet/entities_fixture.jsonl')
        }

        fake_entity_world = {"Germany":"11867", "United_Kingdom": "31717", "European_Commission": "42336"}
        candidate_generator = WikiCandidateMentionGenerator('tests/fixtures/linking/priors.txt',
                                                            entity_world_path=fake_entity_world)
        train_file = 'tests/fixtures/linking/aida.txt'

        reader = LinkingReader(mention_generator=candidate_generator,
                               entity_indexer=_get_indexer("entity_wiki"),
                               extra_candidate_generators=extra_generator)
        instances = reader.read(train_file)

        assert len(instances) == 2


    def test_wiki_linking_reader(self):

        fake_entity_world = {"Germany":"11867", "United_Kingdom": "31717", "European_Commission": "42336"}
        candidate_generator = WikiCandidateMentionGenerator('tests/fixtures/linking/priors.txt',
                                                            entity_world_path=fake_entity_world)
        train_file = 'tests/fixtures/linking/aida.txt'

        reader = LinkingReader(mention_generator=candidate_generator)
        instances = reader.read(train_file)

        instances = list(instances)

        fields = instances[0].fields

        text = [x.text for x in fields["tokens"].tokens]
        assert text == ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']

        spans = fields["candidate_spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])
        assert span_starts == (6, 2)
        assert span_ends == (6, 2)
        gold_ids = [x.text for x in fields["gold_entities"].tokens]
        assert gold_ids == ['United_Kingdom', 'Germany']

        candidate_token_list = [x.text for x in fields["candidate_entities"].tokens]
        candidate_tokens = []
        for x in candidate_token_list:
            candidate_tokens.extend(x.split(" "))

        assert  candidate_tokens == ['United_Kingdom', 'Germany']

        numpy.testing.assert_array_almost_equal(fields["candidate_entity_prior"].array, numpy.array([[1.], [1.]]))
        fields = instances[1].fields
        text = [x.text for x in fields["tokens"].tokens]
        assert text ==['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed',
                       'with', 'German', 'advice', 'to', 'consumers', 'to', 'shun', 'British', 'lamb',
                       'until', 'scientists', 'determine', 'whether', 'it', 'is', 'dangerous']

        spans = fields["candidate_spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])
        assert span_starts == (15, 9)
        assert span_ends == (15, 9)
        gold_ids = [x.text for x in fields["gold_entities"].tokens]
        # id not inside our mini world, should be ignored
        assert "European_Commission" not in gold_ids
        assert gold_ids == ['United_Kingdom', 'Germany']
        candidate_token_list = [x.text for x in fields["candidate_entities"].tokens]
        candidate_tokens = []
        for x in candidate_token_list:
            candidate_tokens.extend(x.split(" "))
        assert candidate_tokens == ['United_Kingdom', 'Germany']

        numpy.testing.assert_array_almost_equal(fields["candidate_entity_prior"].array, numpy.array([[1.], [1.]]))

class TestWikiCandidateMentionGenerator(AllenNlpTestCase):
    def test_wiki_candidate_generator_no_candidates(self):
        fake_entity_world = {"Germany":"11867", "United_Kingdom": "31717", "European_Commission": "42336"}

        candidate_generator = WikiCandidateMentionGenerator(
            'tests/fixtures/linking/priors.txt',
            entity_world_path=fake_entity_world
        )

        candidates = candidate_generator.get_mentions_raw_text(".")
        assert candidates['candidate_entities'] == [['@@PADDING@@']]

    def test_wiki_candidate_generator_simple(self):
        candidate_generator = WikiCandidateMentionGenerator(
            'tests/fixtures/linking/priors.txt',
        )
        s = "Mexico is bordered to the north by the United States."

        # first candidate in each list
        candidates = candidate_generator.get_mentions_raw_text(s)
        first_prior = [span_candidates[0] for span_candidates in candidates['candidate_entities']]
        assert first_prior == ['Mexico', 'United_States']

        # now do it randomly
        candidate_generator.random_candidates = True
        candidate_generator.p_e_m_keys_for_sampling = list(candidate_generator.p_e_m.keys())
        candidates = candidate_generator.get_mentions_raw_text(s)
        first_prior = [span_candidates[0] for span_candidates in candidates['candidate_entities']]
        assert first_prior != ['Mexico', 'United_States']
