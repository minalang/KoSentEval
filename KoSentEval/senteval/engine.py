# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.sts import  KorSTSEval
from senteval.probing import *

class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['KorSTS','Length', 'WordContent', 'Depth','TopConstituents', 'SubjOmission',
                           'Tense', 'Honorifics', 'Sentiment', 'SentType']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)

        # Original SentEval tasks
        if name == 'KorSTS': #STS과제
            self.evaluation = eval(name + 'Eval')(tpath + '/sts/', seed=self.params.seed) # + fpath

        # Probing Tasks
        ## surface level
        elif name == 'Length':
                self.evaluation = LengthEval(tpath + '/korean_probing', seed=self.params.seed)
        elif name == 'WordContent':
                self.evaluation = WordContentEval(tpath + '/korean_probing', seed=self.params.seed)
                
        ## syntactic level
        elif name == 'TopConstituents':
                self.evaluation = TopConstituentsEval(tpath + '/korean_probing', seed=self.params.seed)
        elif name == 'SubjOmission':
                self.evaluation = SubjOmissionEval(tpath + '/korean_probing', seed=self.params.seed)
        
        ## semantic level
        elif name == 'Tense':
                self.evaluation = TenseEval(tpath + '/korean_probing', seed=self.params.seed)
        elif name == 'Honorifics':
                self.evaluation = HonorificsEval(tpath + '/korean_probing', seed=self.params.seed)
        elif name == 'Sentiment':
                self.evaluation = SentimentEval(tpath + '/korean_probing', seed=self.params.seed)
        elif name == 'SentType':
                self.evaluation = SentTypeEval(tpath + '/korean_probing', seed=self.params.seed)

        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
