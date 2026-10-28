import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import contextlib
import io

from d3study.core import COMPLETE, StudyError, atomic_json, digest, event, freeze_run, load_config, parse_action, read_json
from d3study.workflow import _write_dataset, _fit_history, _evidence, generate_attempt, generate, select, check_decisions, forbidden_paths
from d3study.reports import analyze, evaluate

ROOT=Path(__file__).resolve().parents[1]


class FakeGenerator:
    def __init__(self, responses=None):
        self.responses=iter(responses) if responses else None
        self.calls=0
    def verify(self): return {'model':'fake','verified':True}
    def count_tokens(self,messages,timeout=30): return 50
    def generate(self,messages,seed,max_new_tokens,timeout):
        self.calls+=1
        content=next(self.responses) if self.responses else f'```bash\n{COMPLETE}\n```'
        return {'content':content,'reasoning':'Do not execute this example: ```bash\nrm -rf /\n```','input_tokens':50,'output_tokens':20,'reasoning_tokens':10,'finish_reason':'stop','raw_response':{},'seconds':.1}


class FakeEnvironment:
    next_id=0
    calls=[]
    def __init__(self,*args,**kwargs):
        self.index=FakeEnvironment.next_id
        FakeEnvironment.next_id+=1
    def execute(self,command,timeout):
        FakeEnvironment.calls.append(command)
        return {'returncode':0,'output':'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n' if command==COMPLETE else 'observed\n'}
    def extract(self):
        return f'diff --git a/source.py b/source.py\n+value = {self.index}\n',['source.py']
    def close(self): pass


class FakeSelector:
    def load(self): return {'model':'fake','device':'cuda:0'}
    def choose(self,state,criteria,instructions):
        assert 'generator' not in state
        assert 'gold' not in json.dumps(state)
        return {'choice':'option_1','seconds':.1}
    def close(self): pass


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.workspace=Path(self.tmp.name)
        self.cfg=json.loads((ROOT/'configs/gpu.example.json').read_text(encoding='utf-8'))
        self.cfg['execution_enabled']=True
        self.cfg['runtime']['debug_tasks']=1
        self.row={'instance_id':'example__repo-1','repo':'example/repo','base_commit':'a'*40,'problem_statement':'Fix the source bug.','patch':'private gold','test_patch':'private tests','FAIL_TO_PASS':'secret'}
        _write_dataset(self.cfg,self.workspace,'debug',[self.row],[],{'kind':'fixture'})
        atomic_json(self.workspace/'data/debug/images.json',{self.row['instance_id']:{'tag':'fake','id':'sha256:fake'}})
        FakeEnvironment.next_id=0
        FakeEnvironment.calls=[]
    def tearDown(self): self.tmp.cleanup()

    def test_config_portable_and_actions_strict(self):
        cfg,workspace=load_config(ROOT/'configs/gpu.example.json')
        self.assertFalse(cfg['execution_enabled'])
        self.assertEqual(workspace,ROOT/'work')
        self.assertIsNone(parse_action('reasoning\n```bash\npwd\n```'))
        self.assertIsNone(parse_action('```bash\npwd\n```\n```bash\nls\n```'))

    def test_public_manifest_has_no_gold_fields(self):
        public=read_json(self.workspace/'data/debug/public.json')[0]
        self.assertEqual(set(public),{'instance_id','repo','base_commit','problem_statement','image_tag'})
        self.assertIn('patch',read_json(self.workspace/'data/debug/evaluation.json')[0])

    def test_config_rejects_candidate_counts_other_than_three(self):
        path=self.workspace/'invalid-config.json'
        for n in (1,2,4,3.0,True):
            with self.subTest(n=n):
                cfg=copy.deepcopy(self.cfg)
                cfg['candidate_pool']['main_n']=n
                atomic_json(path,cfg)
                with self.assertRaisesRegex(StudyError,'exactly three'):
                    load_config(path)

    def test_debug_excludes_verified_tasks(self):
        with self.assertRaises(StudyError):
            _write_dataset(self.cfg,self.workspace,'debug',[self.row],[self.row['instance_id']],{})

    def test_installed_console_entrypoint_returns_success(self):
        from d3study.cli import main
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', str(ROOT/'configs/gpu.example.json'), 'validate-config']), 0)

    def test_server_uses_configured_ollama_executable(self):
        from d3study.cli import serve
        process=SimpleNamespace(pid=123,wait=lambda:0,poll=lambda:0)
        self.cfg['runtime']['ollama']['cuda_compat_dir']='/opt/cuda-compat'
        seen={}
        def inventory(strict=False):
            seen['ld']=os.environ.get('LD_LIBRARY_PATH','')
            return {'devices':[{}]}
        previous=os.environ.get('LD_LIBRARY_PATH')
        try:
            with patch('d3study.cli.require_linux'), patch('d3study.device.cuda_inventory',inventory), patch('d3study.cli.subprocess.Popen',return_value=process) as launch:
                serve(self.cfg,self.workspace,'generator_small')
        finally:
            if previous is None:
                os.environ.pop('LD_LIBRARY_PATH', None)
            else:
                os.environ['LD_LIBRARY_PATH']=previous
        self.assertEqual(launch.call_args.args[0],['/opt/ollama/bin/ollama','serve'])
        self.assertEqual(launch.call_args.kwargs['env']['OLLAMA_MODELS'],'/opt/ollama/models')
        self.assertEqual(launch.call_args.kwargs['env']['OLLAMA_CONTEXT_LENGTH'],'262144')
        expected='/opt/cuda-compat'+os.pathsep+'/opt/ollama/lib/ollama'
        self.assertTrue(seen['ld']==expected or seen['ld'].startswith(expected+os.pathsep))
        self.assertEqual(launch.call_args.kwargs['env']['LD_LIBRARY_PATH'], seen['ld'])
        deployment=read_json(self.workspace/'services/generator_small/deployment.json')
        self.assertEqual(deployment['server_pid'],123)
        self.assertEqual(deployment['ollama_model'],'qwen3.5:2b')

    def test_generation_rejects_unmanaged_endpoint_before_calls(self):
        with patch('d3study.workflow.require_linux'), self.assertRaises(StudyError):
            generate(self.cfg,self.workspace,'debug','generator_small')

    def test_sampling_change_rejected_after_preparation(self):
        self.cfg['dataset']['sampling_seed'] += 1
        with self.assertRaises(StudyError):
            freeze_run(self.cfg,self.workspace,'debug')

    def test_test_and_evaluation_changes_rejected(self):
        self.assertEqual(forbidden_paths(['tests.py','test.py','pytest.ini','src/test_bug.py','src/source.py']), ['tests.py','test.py','pytest.ini','src/test_bug.py'])

    def test_trusted_extraction_ignores_mutable_shell_startup(self):
        from d3study.docker_env import DockerSession
        session=object.__new__(DockerSession)
        session.container='fake'; session.exe='docker'; session.base='a'*40
        with patch('d3study.docker_env.subprocess.run', return_value=SimpleNamespace(returncode=0,stdout=b'')) as run:
            session.extract()
        self.assertEqual(run.call_count,3)
        for call in run.call_args_list:
            command=call.args[0]
            self.assertIn('BASH_ENV=',command)
            self.assertNotIn('BASH_ENV=/root/.bashrc',command)
            self.assertIn('--noprofile',command)

    def test_completed_attempt_promoted_without_resampling(self):
        run=freeze_run(self.cfg,self.workspace,'debug')
        target=run/'pools/generator_small'/self.row['instance_id']/'candidate_0'
        generate_attempt(self.cfg,self.row,'generator_small',0,{'id':'fake'},target/'attempt_0001',FakeGenerator(),FakeEnvironment)
        event(target/'attempt_0002/events.jsonl', {'type':'execution','command':'WRONG_ATTEMPT','observation':{'returncode':0,'output':'wrong'}})
        backend=FakeGenerator()
        with patch('d3study.workflow.require_linux'), contextlib.redirect_stdout(io.StringIO()):
            generate(self.cfg,self.workspace,'debug','generator_small',backend,FakeEnvironment)
        self.assertEqual(backend.calls,2)
        self.assertEqual(read_json(target/'result.json')['accepted_attempt'],'attempt_0001')
        self.assertNotIn('WRONG_ATTEMPT',json.dumps(_evidence(run,'generator_small',self.row['instance_id'],0,1000)))

    def test_failed_evaluation_retries_only_missing_reports(self):
        with patch('d3study.workflow.require_linux'), contextlib.redirect_stdout(io.StringIO()):
            for gen in ('generator_small','generator_large'):
                generate(self.cfg,self.workspace,'debug',gen,FakeGenerator(),FakeEnvironment)
            for judge in ('decision_small','decision_large'):
                select(self.cfg,self.workspace,'debug',judge,FakeSelector())
        calls=[]
        fail_once=[True]
        def harness(command,cwd,**kwargs):
            calls.append(command)
            if fail_once[0]:
                fail_once[0]=False
                return SimpleNamespace(returncode=1)
            predictions=read_json(command[command.index('--predictions_path')+1])
            run_id=command[command.index('--run_id')+1]
            for p in predictions:
                path=cwd/'logs/run_evaluation'/run_id/p['model_name_or_path']/p['instance_id']/'report.json'
                atomic_json(path,{p['instance_id']:{'resolved':True}})
            return SimpleNamespace(returncode=0)
        with patch('d3study.reports.require_linux'), patch('d3study.reports.inspect_image',return_value={'Id':'sha256:fake'}), patch('d3study.reports.subprocess.run',side_effect=harness):
            self.assertEqual(evaluate(self.cfg,self.workspace,'debug')['evaluation_errors'],1)
            self.assertEqual(len(calls),6)
            self.assertEqual(evaluate(self.cfg,self.workspace,'debug')['evaluation_errors'],0)
            self.assertEqual(len(calls),7)
            self.assertTrue(calls[-1][calls[-1].index('--run_id')+1].endswith('-a2'))
            evaluate(self.cfg,self.workspace,'debug')
            self.assertEqual(len(calls),7)
        run=self.workspace/'runs/debug'
        event(run/'pools/generator_small'/self.row['instance_id']/'candidate_0/attempt_0099/events.jsonl',{'type':'model_reservation','step':0,'input_tokens_counted':123,'output_tokens_upper_bound':456})
        report=analyze(self.cfg,self.workspace,'debug')
        self.assertEqual(report['measured_costs']['unresolved_model_requests'],1)
        self.assertEqual(report['measured_costs']['unresolved_reserved_output_token_upper_bound'],456)
        self.assertEqual(report['generators']['generator_small']['selectors']['decision_small']['valid_decision_count'],1)

    def test_never_executes_reasoning(self):
        backend=FakeGenerator(['```bash\npwd\n```',f'```bash\n{COMPLETE}\n```'])
        result=generate_attempt(self.cfg,self.row,'generator_large',0,{'id':'fake'},self.workspace/'one',backend,FakeEnvironment)
        self.assertEqual(result['termination_reason'],'submitted')
        self.assertEqual(FakeEnvironment.calls,['pwd',COMPLETE])
        self.assertEqual(result['stats']['output_tokens'],40)

    def test_length_truncation_does_not_execute(self):
        backend=FakeGenerator()
        original=backend.generate
        def truncated(*args,**kwargs):
            data=original(*args,**kwargs)
            data['content']=''
            data['finish_reason']='length'
            return data
        backend.generate=truncated
        result=generate_attempt(self.cfg,self.row,'generator_small',0,{'id':'fake'},self.workspace/'short',backend,FakeEnvironment)
        self.assertEqual(result['termination_reason'],'generation_truncated_before_action')
        self.assertEqual(FakeEnvironment.calls,[])
        self.assertEqual(result['stats']['model_calls'],3)

    def test_empty_length_reply_can_continue_into_a_patch(self):
        steps=iter([
            {'content':'','reasoning':'loop','input_tokens':10,'output_tokens':20,'reasoning_tokens':20,'finish_reason':'length','raw_response':{},'seconds':.1},
            {'content':'```bash\nprintf repaired > source.py\n```','reasoning':'edit','input_tokens':12,'output_tokens':8,'reasoning_tokens':2,'finish_reason':'stop','raw_response':{},'seconds':.1},
            {'content':f'```bash\n{COMPLETE}\n```','reasoning':'submit','input_tokens':14,'output_tokens':6,'reasoning_tokens':1,'finish_reason':'stop','raw_response':{},'seconds':.1},
        ])
        backend=FakeGenerator()
        backend.generate=lambda *args, **kwargs: next(steps)
        result=generate_attempt(self.cfg,self.row,'generator_small',0,{'id':'fake'},self.workspace/'recover',backend,FakeEnvironment)
        self.assertEqual(result['termination_reason'],'submitted')
        self.assertEqual(FakeEnvironment.calls,['printf repaired > source.py', COMPLETE])
        self.assertTrue(result['model_patch'])

    def test_complete_shared_pipeline_with_mocked_models(self):
        with patch('d3study.workflow.require_linux'):
            for role in ('generator_small','generator_large'):
                generator=FakeGenerator()
                generate(self.cfg,self.workspace,'debug',role,generator,FakeEnvironment)
                self.assertEqual(generator.calls,3)
                generate(self.cfg,self.workspace,'debug',role,generator,FakeEnvironment)
                self.assertEqual(generator.calls,3)  # resume doesn't resample sealed slots
            for judge in ('decision_small','decision_large'):
                select(self.cfg,self.workspace,'debug',judge,FakeSelector())
        run=self.workspace/'runs/debug'
        decisions=[read_json(p) for p in (run/'decisions').rglob('*.json')]
        self.assertEqual(len(decisions),4)
        for gen in ('generator_small','generator_large'):
            matches=[d for d in decisions if d['generator']==gen]
            self.assertEqual(matches[0]['display_to_candidate'],matches[1]['display_to_candidate'])
            self.assertEqual(matches[0]['pool_sha256'],matches[1]['pool_sha256'])
        check_decisions(self.cfg,self.workspace,'debug')
        labels={}
        for gen in ('generator_small','generator_large'):
            for i in range(3):
                p=read_json(run/'pools'/gen/self.row['instance_id']/f'candidate_{i}/result.json')
                labels[f'{gen}/{self.row["instance_id"]}/{i}']={'resolved':i==0,'patch_sha256':p['patch_sha256']}
        atomic_json(run/'evaluation/labels.json',labels)
        atomic_json(run/'evaluation/decision_seal.json',{str(p.relative_to(run)):digest(read_json(p)) for p in (run/'decisions').rglob('*.json')})
        result=analyze(self.cfg,self.workspace,'debug')
        self.assertEqual(result['generators']['generator_small']['oracle_rate'],1)
        self.assertAlmostEqual(result['generators']['generator_small']['random_expected_rate'],1/3)
        first=next((run/'decisions').rglob('*.json'))
        data=read_json(first); data['selected_patch_sha256']='bad'; atomic_json(first,data)
        with self.assertRaises(StudyError): check_decisions(self.cfg,self.workspace,'debug')

    def test_context_trimming_removes_whole_pairs(self):
        backend=FakeGenerator()
        backend.count_tokens=lambda messages,timeout=30:len(messages)*10
        messages=[{'role':'user','content':'task'}, {'role':'assistant','content':'old'}, {'role':'user','content':'old observation'}, {'role':'assistant','content':'new'}, {'role':'user','content':'new observation'}]
        import time
        fitted,count=_fit_history(backend,messages,40,time.monotonic()+10)
        self.assertEqual([m['role'] for m in fitted],['user','assistant','user'])
        self.assertEqual(fitted[1]['content'],'new')
        self.assertEqual(count,30)

    def test_history_kept_when_prompt_fits_context(self):
        backend=FakeGenerator()
        backend.count_tokens=lambda messages,timeout=30:len(messages)*10
        messages=[{'role':'user','content':'task'}, {'role':'assistant','content':'pwd'}, {'role':'user','content':'OBSERVATION:\nexit 1'}]
        import time
        fitted,count=_fit_history(backend,messages,262144,time.monotonic()+10)
        self.assertEqual([m['content'] for m in fitted],[m['content'] for m in messages])
        self.assertEqual(count,30)

    def test_second_turn_sees_observation(self):
        seen=[]
        class Recording(FakeGenerator):
            def count_tokens(self,messages,timeout=30):
                return 10*len(messages)
            def generate(self,messages,seed,max_new_tokens,timeout):
                seen.append((list(messages), max_new_tokens))
                self.calls+=1
                content='```bash\npwd\n```' if self.calls==1 else f'```bash\n{COMPLETE}\n```'
                return {'content':content,'reasoning':'trace','input_tokens':10*len(messages),'output_tokens':20,'reasoning_tokens':1,'finish_reason':'stop','raw_response':{},'seconds':.1}
        result=generate_attempt(self.cfg,self.row,'generator_small',0,{'id':'fake'},self.workspace/'kept',Recording(),FakeEnvironment)
        self.assertEqual(result['termination_reason'],'submitted')
        self.assertGreaterEqual(len(seen[1][0]),3)
        self.assertIn('OBSERVATION:', seen[1][0][-1]['content'])
        self.assertEqual(seen[1][1], self.cfg['rollout_limits']['generator_max_context_tokens']-len(seen[1][0])*10)


if __name__=='__main__': unittest.main()
