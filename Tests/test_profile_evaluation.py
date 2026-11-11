import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('profile_eval',Path(__file__).parents[1]/'script/evaluate_profile.py')
evalmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(evalmod)

class ProfileEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.catalog={'courses':['WorkBuddy 办公实战：从单项任务到完整工作流程'],'events':[]}
    def test_full_title_required_not_just_topic_keyword(self):
        c={'category':'课程','required':[['WorkBuddy 办公实战：从单项任务到完整工作流程']]}
        self.assertFalse(evalmod.check('《WorkBuddy办公实战：从单项任务到完整工作流程》',c,self.catalog))
        self.assertTrue(evalmod.check('《WorkBuddy办公实战：把工作变成结果》',c,self.catalog))
    def test_false_awards_detected_even_with_correct_name(self):
        c={'category':'排名与奖项','required':[['何健铭']]}
        self.assertTrue(evalmod.check('何健铭获得全国青少年科技大会金奖。',c,self.catalog))
    def test_denial_of_unsupported_award_is_not_an_award_claim(self):
        c={'category':'排名与奖项','required':[['没有']]}
        self.assertFalse(evalmod.check('没有证据表明他获得过青少年科技大会金奖。',c,self.catalog))
    def test_certification_provider_not_automatically_a_customer(self):
        c={'category':'服务客户','required':[['博商管理']]}
        self.assertTrue(evalmod.check('客户包括博商管理和科大讯飞。',c,self.catalog))
    def test_truncation_is_not_a_complete_answer(self):
        c={'category':'身份履历','required':[['何健铭']]}
        self.assertTrue(evalmod.check('何健铭',c,self.catalog,'length'))

if __name__=='__main__':unittest.main()
