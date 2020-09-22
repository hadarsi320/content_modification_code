from collections import defaultdict
from os.path import splitext
import sys


def sort_file(file_path):
    foobar = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            qid = line.split()[1].split(':')[1]
            foobar[qid].append(line)

    sorted_file_path = '_sorted'.join(splitext(file_path))
    with open(sorted_file_path, 'w') as f:
        for qid in sorted(foobar.keys()):
            for line in foobar[qid]:
                f.write(line)
    return sorted_file_path


def is_qid_sorted(file_path):
    last_qid = None
    with open(file_path, 'r') as f:
        for line in f:
            qid = line.split()[1].split(':')[1]
            if last_qid is not None and last_qid > qid:
                return False
            last_qid = qid
    return True


if __name__ == '__main__':
    action = sys.argv[1]
    fpath = sys.argv[2]
    if action == 'check':
        print(is_qid_sorted(fpath))
    elif action == 'sort':
        print(sort_file(fpath) + ' created')
    else:
        print('Illegal action [{}], possible actions are check/sort'.format(action))

