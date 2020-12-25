import math
import kaldi.fstext as fst

class SortedMatcher(object):
    """
    class implements searching arc/scores on FST
    
    Args:
        vector_fst (object): loaded fst
        max_num_arcs (int): maximum number of arcs starting from one fst state
        max_id (int): maximum i/o label id of LM fst
        backoff_id (int): backoff id of LM fst
        disambig_ids (List of int): disambig ids of LM fst
    """
    def __init__(self, vector_fst, max_num_arcs,
                 max_id, backoff_id, disambig_ids):
        #make sure fst is i/o label sorted
        self.fst = vector_fst
        self.max_num_arcs = max_num_arcs
        self.max_id = max_id
        self.backoff_id = backoff_id
        self.disambig_ids = disambig_ids

    def search(self, state_id, ilabel):
        """
        binary search on ArcIterator
        """
        aiter = self.fst.arcs(state_id)
        #binary search on ArcIterator
        size = self.max_num_arcs
        high = size - 1
        while size > 1:
            half = size // 2
            mid = high - half
            aiter.seek(mid)
            if aiter.done():
                cur_id = self.max_id
            else:
                cur_id = aiter.value().ilabel
            if cur_id >= ilabel:
                high = mid
            size -= half
        aiter.seek(high)
        if aiter.done():
            return False, None
        if aiter.value().ilabel == ilabel:
            return True, aiter
        return False, None

    def get_scores_wodisambig(self, state_id, ilabel, init_score=0.0):
        scores = []
        states = []
        bf_score = init_score
        cur_state = state_id
        while True:
            has_arc, aiter = self.search(cur_state, ilabel)
            if has_arc:
                scores.append(bf_score + aiter.value().weight.value)
                states.append(aiter.value().nextstate)
            has_backoff, aiter_bf = self.search(cur_state, self.backoff_id)
            if has_backoff:
                bf_score += aiter_bf.value().weight.value
                cur_state = aiter_bf.value().nextstate
            else:
                return scores, states

    def get_scores(self, state_id, ilabel):
        init_scores = [0.0]
        init_states = [state_id]
        #check disambig arcs,
        for label in self.disambig_ids:
            found, aiter = self.search(state_id, label)
            if found:
                init_scores.append(aiter.value().weight.value)
                init_states.append(aiter.value().nextstate)
        scores, states = [], []
        for i, init_score in enumerate(init_scores):
            cur_sc, cur_st = self.get_scores_wodisambig(init_states[i],
                                                        ilabel, init_score)
            scores.extend(cur_sc)
            states.extend(cur_st)
        return scores, states

    def final_score(self, state_id):
        final_scores = [0.0]
        final_states = [state_id]
        #check disambig arcs,
        for label in self.disambig_ids:
            found, aiter = self.search(state_id, label)
            if found:
                final_scores.append(aiter.value().weight.value)
                final_states.append(aiter.value().nextstate)
        def search_final(state_id, init_score=0.0):
            score = init_score
            cur_state = state_id
            while True:
                final_score = self.fst.final(cur_state).value
                if math.isinf(final_score):
                    found, aiter = self.search(cur_state, self.backoff_id)
                    if found:
                        score += aiter.value().weight.value
                        cur_state = aiter.value().nextstate
                    else:
                        return float('inf'), None
                else:
                    score += final_score
                    return score, cur_state
        for i, final_score in enumerate(final_scores):
            final_scores[i], final_states[i] = search_final(final_states[i],
                                                            final_score)
        return final_scores, final_states
