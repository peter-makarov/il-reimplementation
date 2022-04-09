"""Unit tests for vocabulary.py."""
import unittest

from trans import vocabulary
from trans.actions import BeginOfSequence, ConditionalDel, ConditionalCopy, \
    ConditionalIns, ConditionalSub, EndOfSequence
from trans.vocabulary import UNK_CHAR, PAD_CHAR


class VocabularyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = vocabulary.Vocabulary()
        for c in "foo":
            cls.vocabulary.encode(c)
        cls.vocabularies = vocabulary.Vocabularies()
        cls.vocabularies.encode_input("foo")
        cls.vocabularies.encode_actions("baa")

    def test_vocabulary(self):
        i2w1 = []
        i2w2 = [4, 5, 6]
        vocabulary1 = vocabulary.Vocabulary(i2w1)
        self.assertListEqual(
            [BeginOfSequence(), EndOfSequence(), PAD_CHAR, UNK_CHAR],
            vocabulary1.i2w)
        self.assertDictEqual(
            {BeginOfSequence(): 0, EndOfSequence(): 1, PAD_CHAR: 2,
             UNK_CHAR: 3}, vocabulary1.w2i)
        vocabulary2 = vocabulary.Vocabulary(i2w2)
        self.assertListEqual(
            [BeginOfSequence(), EndOfSequence(), PAD_CHAR,
             UNK_CHAR, 4, 5, 6],
            vocabulary2.i2w)
        self.assertDictEqual(
            {BeginOfSequence(): 0, EndOfSequence(): 1, PAD_CHAR: 2,
             UNK_CHAR: 3, 4: 4, 5: 5, 6: 6},
            vocabulary2.w2i)

    def test_actions(self):
        vocabulary1 = vocabulary.ActionVocabulary()
        expected_i2w = [BeginOfSequence(), EndOfSequence(),
                        PAD_CHAR, ConditionalDel(), ConditionalCopy()]
        expected_w2i = {BeginOfSequence(): 0, EndOfSequence(): 1,
                        PAD_CHAR: 2, ConditionalDel(): 3, ConditionalCopy(): 4}
        self.assertListEqual(expected_i2w, vocabulary1.i2w)
        self.assertDictEqual(expected_w2i, vocabulary1.w2i)

    def test_vocabulary_encode(self):
        vocabulary1 = vocabulary.Vocabulary()
        for c in "foo":
            vocabulary1.encode(c)
        self.assertEqual(6, len(vocabulary1))
        self.assertEqual(5, vocabulary1.encode("o"))

    def test_vocabulary_decode(self):
        self.assertEqual("f", self.vocabulary.decode(4))
        self.assertRaises(IndexError, self.vocabulary.decode, 10)

    def test_vocabulary_lookup(self):
        self.assertEqual(3, self.vocabulary.lookup("F"))

    def test_vocabularies_encode_input(self):
        vocabulary1 = vocabulary.Vocabularies()
        encoded_foo = vocabulary1.encode_input("foo")
        self.assertListEqual([0, 4, 5, 5, 1], encoded_foo)

    def test_vocabularies_encode_actions(self):
        vocabulary1 = vocabulary.Vocabularies()
        vocabulary1.encode_actions("baa")
        expected_i2w = [BeginOfSequence(), EndOfSequence(),
                        PAD_CHAR, ConditionalDel(),
                        ConditionalCopy(), ConditionalSub("b"),
                        ConditionalIns("b"), ConditionalSub("a"),
                        ConditionalIns("a")]
        self.assertListEqual(expected_i2w, vocabulary1.actions.i2w)

    def test_vocabularies_encode_unseen_input(self):
        encoded_fox = self.vocabularies.encode_unseen_input("fox")
        self.assertListEqual([0, 4, 5, 3, 1], encoded_fox)

    def test_vocabularies_encode_unseen_actions(self):
        encoded_action = self.vocabularies.encode_unseen_action(
            ConditionalIns("b"))
        self.assertEqual(6, encoded_action)
        self.assertRaises(KeyError, self.vocabularies.encode_unseen_action,
                          ConditionalIns("Q"))

    def test_vocabularies_decode_actions(self):
        decoded_actions = [
            self.vocabularies.decode_action(i) for i in (0, 4, 5, 2)]
        expected_actions = [BeginOfSequence(), ConditionalCopy(),
                            ConditionalSub("b"), PAD_CHAR]
        self.assertListEqual(expected_actions, decoded_actions)
        self.assertRaises(IndexError, self.vocabularies.decode_action, 10)

    def test_vocabularies_decode_input(self):
        expected_input = self.vocabularies.decode_input([5, 5, 4])
        self.assertEqual("oof", expected_input)
        self.assertRaises(IndexError, self.vocabularies.decode_input, [6])


if __name__ == "__main__":
    VocabularyTests().run()
