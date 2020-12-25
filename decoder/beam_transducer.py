"""
Class for managing the internals of the beam search process.
Takes care of beams, back pointers, and scores.
"""
from __future__ import division
import copy
from collections import defaultdict
import torch

class BeamMergeTransducer():
    """
    Args:
        size (int): beam size
        pad, bos, eos (int): indices of padding, beginning, and ending.
        n_best (int): nbest size to use
        cuda (bool): use gpu
        global_scorer (:obj:`GlobalScorer`)
        beam_prune (bool): prune beam that has redundant partial hyp 
    """
    def __init__(self, size, blk=0,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 max_len=10000,
                 lm_scorer=None,
                 lm_scorer_scale=1.0, 
                 beam_prune=True,
                 args=None):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(blk)]
        self.next_ys[0][0] = blk
        self.blk = blk
        # Has EOS topped the beam yet.
        self._eos = -1
        self.eos_top = False
        #partial hypothesis
        self.prev_part_hyp = [[] for _ in range(self.size)]
        self.cur_part_hyp = [[] for _ in range(self.size)]
        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best
        self.beam_prune = beam_prune
        self.args = args

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        self.max_len = max_len

        self.lm_scorer = lm_scorer
        self.lm_scorer_scale = lm_scorer_scale
        # list of set to keep active
        # FST states for each beam
        self.state_sets = [defaultdict(lambda: float('inf')) \
                           for _ in range(self.size)]
        for state_map in self.state_sets:
            state_map[0] = 0.0
        self.lm_scores = self.tt.FloatTensor(size).zero_()
        #self.global_lm_score = [0.0] * self.size


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, t_idx, num_frames):
        """
        Given prob over words for every last beam

        Args:
        word_probs - probs of advancing from the last step (K x words)

        """
        num_words = word_probs.size(1)

        # Sum the previous scores.
        if self.prev_ks:
            beam_scores = word_probs + \
                          self.scores.unsqueeze(1).expand_as(word_probs) + \
                          (self.lm_scorer_scale * \
                          self.lm_scores.unsqueeze(1).expand_as(word_probs))

            # Don't let finished beam have children.
            # merge partial hyps that have same sequence.
            part_hyp_map = dict()
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
                #Beam Prune#
                elif self.beam_prune:
                    part_hyp = str(self.get_current_hyp(i))
                    #empty list is len('[]') == 2
                    if len(part_hyp) > 2:
                        if part_hyp in part_hyp_map:
                            #disable redundant beam
                            beam_scores[i] = -1e20
                        else:
                            part_hyp_map[part_hyp] = i
            #swap cur and prev partial hyps
            self.prev_part_hyp = copy.deepcopy(self.cur_part_hyp)
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        self.all_scores.append(self.scores)
        #we need substract on-the-fly lm scores here
        self.scores = best_scores
        self.scores -= self.lm_scorer_scale * self.lm_scores[prev_k]


        #on-the-fly rescore with fst
        if self.lm_scorer:
            next_state_sets = [defaultdict(lambda: float('inf')) \
                               for _ in range(self.size)]
            for i in range(self.next_ys[-1].size(0)):
                ilabel = self.next_ys[-1][i] + 1
                if self.next_ys[-1][i] != self.blk:
                    for state in self.state_sets[prev_k[i]].keys():
                        scores, states = self.lm_scorer.get_scores(state,
                                                                   ilabel)
                        for next_state, cost in zip(states, scores):
                            next_cost = self.state_sets[prev_k[i]][state]
                            next_cost += cost
                            if next_cost < next_state_sets[i][next_state]:
                                next_state_sets[i][next_state] = \
                                    next_cost - self.args.nonblk_reward
                else:
                    for k, v in self.state_sets[prev_k[i]].items():
                        next_state_sets[i][k] = v
                if next_state_sets[i]:
                    self.lm_scores[i] = -min([v for _, v in \
                                             next_state_sets[i].items()])
                else:
                    self.lm_scores[i] = -1e20
            self.state_sets, next_state_sets = next_state_sets, self.state_sets

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self.blk and \
               t_idx[prev_k[i]].item() == num_frames - 1 or \
               len(self.next_ys) > self.max_len:
                s = self.scores[i]
                self.next_ys[-1][i] = self._eos
                if self.lm_scorer:
                    final_scores = defaultdict(lambda: float('inf'))
                    for state in self.state_sets[i].keys():
                        f_scores, f_states = self.lm_scorer.final_score(state)
                        for f_s, cost in zip(f_states, f_scores):
                            next_cost = self.state_sets[i][state] + cost
                            if next_cost < final_scores[f_s]:
                                final_scores[f_s] = next_cost
                    final_lm_score = -min([v for _, v in final_scores.items()])
                    s += self.lm_scorer_scale * final_lm_score
                    #self.global_lm_score[i] -= min(final_scores)
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
            else:
                self.update_partial_hyp(i)
        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            # self.all_scores.append(self.scores)
            self.eos_top = True


    def done(self):
        """
        if beam search process finishes
        """
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        """
        sort all hypothesis in the finished beams
        Args:
            minimum(int): number of output hypothesis
        """
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def update_partial_hyp(self, k):
        """
        update all partial hypothsis if encounter non-blk
        """
        if k != self.prev_ks[-1][k].item():
            self.cur_part_hyp[k] = \
                copy.deepcopy(self.prev_part_hyp[self.prev_ks[-1][k]])
        y = self.next_ys[-1][k].item()
        if y != self.blk:
            self.cur_part_hyp[k].append(y)

    def get_current_hyp(self, k):
        """
        get current hypothesis for beam k (with blk removed)
        """
        return self.cur_part_hyp[k]

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            #if self.next_ys[j+1][k] != self.blk:
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]


class GlobalScorer():
    """
    Global Rescorer Interface
    Args:

    """
    def __init__(self):
        pass
    def score(self, beam, logprobs):
        "Additional term add to log probability"
        #l_term = len(beam.next_ys) if len(beam.next_ys) > 0 else 0.001
        #return logprobs / l_term
        return logprobs
