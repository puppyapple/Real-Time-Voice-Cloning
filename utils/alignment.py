import textgrid
import glob
from multiprocessing.pool import Pool
from functools import partial
from itertools import chain
from tqdm import tqdm
from pathlib import Path
from os.path import join, exists, basename, splitext

def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""

    ind = 0
    str_interval = str(IntervalObject)
    # print(str_interval)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name

    return {pn: (st, et)}


def parse_textgrid(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList("words")
    words_list = list_words[0]
    result = []
    for ele in words_list:
        d = parse_Interval(ele)
        result.append(list(d.items())[0])
    text, time = zip(*result)
    text = list(text)
    time = [str(t[1]) for t in time]
    name = basename(filename).split('.')[0]
    if text[-1] != 'None':
        text.append('None')
        time.append(time[-1])
    if text[0] != 'None':
        text.insert(0, 'None')
        time.insert(0, time[0])
    to_write = ' '.join([name, ','.join(text), ','.join(time)])
    write_path = filename.parent.joinpath(f'{name}.alignment.txt')
    with open(write_path, 'w') as f:
        f.write(to_write)
        f.close()

        
def prepare_align_dataset(path: Path, n_processes=8, suffix='*.TextGrid'):
    files = list(chain.from_iterable(speaker.glob(suffix) for speaker in path.glob('*')))
    func = partial(parse_textgrid)
    job = Pool(n_processes).imap(func, files)
    list(tqdm(job, "alignment", len(files), unit="utturances"))