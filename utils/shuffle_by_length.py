import argparse
import random

FLAGS = None

def main():
    feats_len_tuples = []
    with open(FLAGS.feats_len) as f:
        for line in f:
            uttid, uttlen = line.split() 
            uttlen = int(uttlen)
            if uttlen <= FLAGS.max_len:
                feats_len_tuples.append((uttid, uttlen))
    feats_len_tuples.sort(key=lambda tup: tup[1], reverse=True)
    #group sorted tuple list
    tuples_batch = []
    if FLAGS.full_batch:
        tuples_len = len(feats_len_tuples) // FLAGS.batch_size * FLAGS.batch_size
    else:
        tuples_len = len(feats_len_tuples)
    for i in range(0, tuples_len, FLAGS.batch_size):
        tuples_batch.append(feats_len_tuples[i:i+FLAGS.batch_size])
    if FLAGS.random:
        random.shuffle(tuples_batch)
    else:
        tuples_batch.reverse()
    with open(FLAGS.feats_len_shuffled, 'w') as f:
        for x in tuples_batch:
            for y in x:
                uttid, uttlen = y
                f.write("{} {}\n".format(uttid, uttlen))  
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'group utterances according to their length and shuffle groups, for sequence or CTC training')
    parser.add_argument('--random', action='store_true', help='shuffle group/batch or the order from short to long')
    parser.add_argument('--batch_size', type=int, default=16, help='group/batch size')
    parser.add_argument('--max_len', type=int, default=3000, help='maximum length, will not take any utterances beyond this length')
    parser.add_argument('--full_batch', action='store_true', help='enable this to discard last batch contains utterances less than batch size')
    parser.add_argument('feats_len', type=str, help='input feats length file') 
    parser.add_argument('feats_len_shuffled', type=str, help='shuffled output feats length file')
    FLAGS, unk = parser.parse_known_args()
    main()    
