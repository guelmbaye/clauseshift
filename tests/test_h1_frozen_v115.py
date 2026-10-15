"""No provider, gold, or final/test fixtures are accessed by these tests."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src import h1_hybrid as h, submission
from src.core import load_jsonl, write_jsonl

class FakeTokenizer:
    def __call__(self, s, **kwargs):
        return {'input_ids': list(range(len(s.split())+2))}
class FakeEncoder:
    def __init__(self):
        self.pipeline=type('Pipeline',(),{'tokenizer':FakeTokenizer()})()
        self.calls=0
    def predict(self,a,b):
        self.calls+=1
        return type('Prediction',(),{
            'tokens_a':tuple(a.split()),'tokens_b':tuple(b.split()),
            'labels_a':tuple(-1.0 if t=='.' else 0.6 for t in a.split()),
            'labels_b':tuple(-1.0 if t=='.' else 0.5 for t in b.split())})()

def fake_api(message,key):
    assert key=='synthetic'
    request=json.loads(message.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
    return json.dumps({'a':{str(i):5 for i in request['requested_a']},
                       'b':{str(i):5 for i in request['requested_b']}})

def setup_inputs(root):
    (root/'data').mkdir()
    for lang in h.LANGS:
        rows=[{'id':f'admin_{lang}_{i}',
               'text_a':f'The document {i} was adopted .',
               'text_b':f'Das Dokument {i} wurde angenommen .'}
              for i in range(18)]
        write_jsonl(root/'data'/f'dev_{lang}.jsonl',rows)
        write_jsonl(root/'data'/f'final_{lang}.jsonl',rows)

class TestH1Frozen(TestCase):
    def test_short_fixture_and_true_response_schema(self):
        row={'id':'admin_de_3','text_a':'The document was adopted .',
             'text_b':'Das Dokument wurde angenommen .',
             'labels_a':[.6,.6,.6,.6,-1], 'labels_b':[.5,.5,.5,.5,-1]}
        spec=h.windows(row)
        parsed=h.parse(fake_api(h.prompt(row,spec),'synthetic'),spec,row)
        fused=h.apply(row,parsed)
        self.assertTrue(fused['labels_a'] != row['labels_a'])
        self.assertEqual(fused['labels_a'][-1],-1)
        with self.assertRaises(h.ContractError):
            h.parse('{"a":{"0":5},"b":{}}',spec,row)
        with self.assertRaises(h.ContractError):
            h.parse('{"a":{},"a":{},"b":{}}',spec,row)

    def test_ablation_engine_checkpoint_and_replay(self):
        with TemporaryDirectory() as td:
            root=Path(td);setup_inputs(root)
            enc=FakeEncoder();calls=[]
            def counted_api(prompt,key):
                calls.append(1)
                return fake_api(prompt,key)
            env={'LLM_NAME':h.MODEL,'LLM_BASE_URL':h.ENDPOINT,'LLM_API_KEY':'synthetic'}
            with patch.dict(os.environ,env):
                with self.assertRaisesRegex(h.ContractError,'incomplete'):
                    h.run_dataset('dev',root/'data',root/'out',root/'state',
                                  'dev:abc123',model_factory=lambda:enc,api=counted_api,max_paid_calls=3)
                self.assertEqual(enc.calls,54)
                self.assertEqual(len(calls),3)
                rows=h.run_dataset('dev',root/'data',root/'out',root/'state',
                                   'dev:abc123',model_factory=lambda:(_ for _ in ()).throw(AssertionError('reloaded encoder')),
                                   api=counted_api)
                self.assertEqual(len(calls),54)
                self.assertEqual(enc.calls,54)
                self.assertEqual(sum(len(v) for v in rows.values()),54)
                rows2=h.run_dataset('dev',root/'data',root/'out',root/'state',
                                    'dev:abc123',model_factory=lambda:(_ for _ in ()).throw(AssertionError('reloaded encoder')),
                                    api=counted_api)
                self.assertEqual(rows2,rows)
                self.assertEqual(len(calls),54)
                for lang in h.LANGS:
                    exported=list(load_jsonl(root/'out'/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'))
                    self.assertEqual([r['id'] for r in exported],[f'admin_{lang}_{i}' for i in range(18)])
                    self.assertTrue(all(len(r['labels_a'])==6 for r in exported))

    def test_fail_closed_gold_in_input(self):
        with TemporaryDirectory() as td:
            root=Path(td);setup_inputs(root)
            p=root/'data'/'dev_de.jsonl';rows=list(load_jsonl(p));rows[0]['labels_a']=[0]*5
            write_jsonl(p,rows)
            with self.assertRaisesRegex(h.ContractError,'only id/text'):
                h.inputs(root/'data','dev')

    def test_final_split_requires_authorization(self):
        with TemporaryDirectory() as td:
            root=Path(td);setup_inputs(root)
            with self.assertRaisesRegex(h.ContractError,'only after frozen'):
                h.run_dataset('final',root/'data',root/'out',root/'state','not_authorized')

    def test_freeze_preflight_h1_and_manifest_tamper(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'manifest.json'
            item=submission.freeze(path,h.UPSTREAM_SHA,'b'*40,h.MODEL,
                         method='H1-HYBRID',embedding_revision=h.HF_SHA)
            self.assertEqual(item['schema'],5)
            self.assertEqual(submission.preflight(path)['method'],'H1-HYBRID')
            item['h1']['blend']=.1
            path.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError,'protocol modified'):
                submission.preflight(path)
