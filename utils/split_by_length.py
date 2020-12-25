import argparse
import random

def main():
    feats_len_tuples = []
    with open(args.feats_len) as f:
        for line in f:
            uttid, uttlen = line.split() 
            uttlen = int(uttlen)
            if uttlen <= args.max_len and uttlen >= args.min_len:
                feats_len_tuples.append((uttid, uttlen))
    feats_len_tuples.sort(key=lambda tup: tup[1], reverse=True)
    #group sorted tuple list
    tuples_batch = []
    block_size = args.batch_size * args.world_size
    if args.full_batch:
        tuples_len = len(feats_len_tuples) // block_size * block_size 
    else:
        tuples_len = len(feats_len_tuples)
    for i in range(0, tuples_len, block_size):
        tuples_batch.append(feats_len_tuples[i:i+block_size])
    if args.random:
        random.shuffle(tuples_batch)
    else:
        tuples_batch.reverse()
    fs = [ open('{}.{}'.format(args.feats_len, i), 'w') for i in range(args.world_size)]
    for x in tuples_batch:
        for i in range(args.world_size):
            for j in range(args.batch_size):
                y = x[i * args.batch_size + j]
                fs[i].write('{} {}\n'.format(y[0], y[1])) 
    for f in fs:
        f.close() 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='split utterances according '
                                                 'to their length and shuffle groups ')
    parser.add_argument('--random', action='store_true', 
                        help='shuffle group/batch or the order from short to long')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='group/batch size')
    parser.add_argument('--world_size', type=int, default=8, 
                        help='number of workers')
    parser.add_argument('--min_len', type=int, default=0, 
                        help='minimum length, discard sequences below this length')
    parser.add_argument('--max_len', type=int, default=3000, 
                        help='maximum length, discard sequences beyond this length')
    parser.add_argument('--full_batch', action='store_true', 
                        help='make sure each batch with bsz')
    parser.add_argument('feats_len', type=str, 
                        help='input feats length file') 
    args, unk = parser.parse_known_args()
    main()    
