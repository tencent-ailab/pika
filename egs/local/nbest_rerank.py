"""
nbest reranking script
"""
import argparse



def main():
    with open(args.in_hyp, 'r', encoding='utf-8') as in_hyp_f,\
         open(args.out_hyp, 'w', encoding='utf-8') as out_hyp_f:
        cur_index = 0
        hyp_score = []
        empty_hyp_len = 1
        for line in in_hyp_f:
            if args.las_rescore:
                empty_hyp_len = 3
            if len(line.split()) <= empty_hyp_len:
                hyp = ''
                score = args.rnnt_score_scale*float(line.split()[0])
                if args.las_rescore:
                    score += args.las_fw_score_scale*float(line.split()[1])
                    score += args.las_bw_score_scale*float(line.split()[2])
            else:
                hyp = line.split()[0]
                hyp = hyp.replace('<unk>', ' ')
                score = args.rnnt_score_scale*float(line.split()[1])
                if args.las_rescore:
                    num_scores = len(line.split()) - 2
                    las_fw_score = sum([float(s) for s in line.split()[2:2+(num_scores)//2]])
                    score += args.las_fw_score_scale * las_fw_score
                    las_bw_score = sum([float(s) for s in line.split()[2+(num_scores)//2:]])
                    score += args.las_bw_score_scale * las_bw_score
            norm = 0.001 if len(hyp) == 0 else len(hyp)
            hyp_score.append((-score/norm, hyp))
            cur_index += 1
            if cur_index == args.nbest:
                hyp_score = sorted(hyp_score, key=lambda x: x[0])
                out_hyp_f.write('{}\n'.format(' '.join([c for c in hyp_score[0][1]])))
                cur_index = 0
                hyp_score = []

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='find oracle from nbest')
    parser.add_argument('in_hyp', type=str,
                        help='input hypothesis, i.e., nbest')
    parser.add_argument('out_hyp', type=str,
                        help='output oracle sequence')
    parser.add_argument('--nbest', type=int,
                        help='number of nbest')
    parser.add_argument('--las_rescore', action='store_true',
                        help='enable las rescore')
    parser.add_argument('--rnnt_score_scale', type=float, default=1.0,
                        help='rnnt score scale')
    parser.add_argument('--las_fw_score_scale', type=float, default=0.3,
                        help='forward las score scale')
    parser.add_argument('--las_bw_score_scale', type=float, default=0.7,
                        help='backward las score scale')
    args, unk = parser.parse_known_args()

    main()
