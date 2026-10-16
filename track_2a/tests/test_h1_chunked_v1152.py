"""Synthetic regression tests only: no DEV, final inputs, models or provider calls."""
from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory

from src import h1_hybrid as h

class CountingTokenizer:
    def __init__(self):
        self.max_seen = 0
    def __call__(self, text, **kw):
        assert kw.get('truncation') is False
        n = len(text.split())+2
        self.max_seen = max(self.max_seen, n)
        return {'input_ids':range(n)}

class TokenEncoder:
    def __init__(self, bad=False):
        self.pipeline = type('Pipe', (), {'tokenizer':CountingTokenizer()})()
        self.calls=[]
        self.bad=bad
    def predict(self, a, b):
        aa=a.split(); bb=b.split()
        self.calls.append((tuple(aa),tuple(bb)))
        if self.bad and len(self.calls)==2:
            aa=aa[1:]
        return type('Pred', (), {
            'tokens_a':tuple(aa),'tokens_b':tuple(bb),
            'labels_a':tuple(int(x[1:])/1000 for x in aa),
            'labels_b':tuple(int(x[1:])/1000 for x in bb)})()

class TestH1SegmentedLongDocuments(TestCase):
    def make_row(self, n=210, m=230):
        return {'id':'admin_de_synthetic',
                'text_a':' '.join('a%03d'%i for i in range(n)),
                'text_b':' '.join('b%03d'%i for i in range(m))}

    def test_short_path_is_single_call_and_same_scores(self):
        encoder=TokenEncoder(); row=self.make_row(12,11)
        scores,chunks=h.infer_doc(encoder,row,90)
        self.assertEqual(chunks,1)
        self.assertEqual(len(encoder.calls),1)
        self.assertEqual(scores['a'],[i/1000 for i in range(12)])
        self.assertEqual(scores['b'],[i/1000 for i in range(11)])

    def test_over_cap_overlap_stitches_exact_original_token_order(self):
        encoder=TokenEncoder(); row=self.make_row()
        scores,chunks=h.infer_doc(encoder,row,90)
        self.assertGreater(chunks,1)
        self.assertLessEqual(chunks,h.MAX_CHUNKS)
        self.assertEqual(len(encoder.calls),chunks)
        self.assertEqual(scores['a'],[i/1000 for i in range(210)])
        self.assertEqual(scores['b'],[i/1000 for i in range(230)])
        self.assertTrue(all(len(a)+2<=90 and len(b)+2<=90 for a,b in encoder.calls))
        self.assertGreater(len(encoder.calls[0][0])+len(encoder.calls[1][0]),210//chunks*2)

    def test_invalid_chunk_fails_before_checkpoint(self):
        encoder=TokenEncoder(bad=True); row=self.make_row()
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(h.ContractError,'coverage/order mismatch'):
                h.predict_all(Path(td),'dev:synthetic',{'de':[row],'fr':[],'it':[]},
                              model_factory=lambda:encoder,max_subtokens=90)
            self.assertFalse(list(Path(td).rglob('*.json')))

    def test_extreme_length_ratio_rejected_without_fabrication(self):
        encoder=TokenEncoder()
        with self.assertRaisesRegex(h.ContractError,'Cannot segment'):
            h.infer_doc(encoder,self.make_row(2,4000),90)
