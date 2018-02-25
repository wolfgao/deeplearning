# -*- coding: UTF-8 -*-

import os

DATADIR = ""
#TODO:
#这里要换上你自己的路径和文件
DATAFILE = "beatles-diskography.csv"

def parse_file(datafile):
    data = []
    with open(datafile, "r") as f:
        titles = f.readline().split(',')

        number = 0


        for line in f:
            if number == 10:
                break
            #already the cursor skip the first line after executing f.readline()
            feilds = {}
            for i in range(len(titles)):
                feilds[titles[i].strip()] = line.split(',')[i].strip()
            data.append(feilds)
            number+=1
    print data[0]
    return data


def test():
    # a simple test of your implemetation
    datafile = os.path.join(DATADIR, DATAFILE)
    d = parse_file(datafile)
    '''
    这里换上你自己的测试数据
    '''
    #firstline = {'Title': 'Please Please Me', 'UK Chart Position': '1', 'Label': 'Parlophone(UK)', 'Released': '22 March 1963', 'US Chart Position': '-', 'RIAA Certification': 'Platinum', 'BPI Certification': 'Gold'}
    #tenthline = {'Title': '', 'UK Chart Position': '1', 'Label': 'Parlophone(UK)', 'Released': '10 July 1964', 'US Chart Position': '-', 'RIAA Certification': '', 'BPI Certification': 'Gold'}

    assert d[0] == firstline
    assert d[9] == tenthline


test()