from __future__ import division
import math
import copy
import random
import datetime
import timeit
import pdb

import sys
from difflib import _calculate_ratio


def main():
    sudogrid = [[0, 3, 0, 5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 4, 0, 8],
                [0, 0, 0, 0, 0, 0, 0, 2, 0],
                [9, 0, 8, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 7, 0],
                [4, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 0, 0, 0, 0, 1, 5, 0],
                [0, 0, 0, 6, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 8, 4, 0, 0, 0]]


    sudogrid3 = [[0, 2, 4, 0, 0, 7, 0, 0, 0],
                [6, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 6, 8, 0, 4, 1, 5],
                [4, 3, 1, 0, 0, 5, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 3, 2],
                [7, 9, 0, 0, 0, 0, 0, 6, 0],
                [2, 0, 9, 7, 1, 0, 8, 0, 0],
                [0, 4, 0, 0, 9, 3, 0, 0, 0],
                [3, 1, 0, 0, 0, 4, 7, 5, 0]]

    sudogridh = [[0, 0, 0, 3, 7, 0, 6, 0, 0],
                [0, 8, 4, 0, 0, 2, 7, 0, 0],
                [2, 0, 0, 0, 1, 0, 0, 4, 0],
                [5, 2, 0, 4, 0, 0, 0, 0, 0],
                [0, 4, 7, 0, 0, 0, 1, 9, 0],
                [0, 0, 0, 0, 0, 1, 0, 7, 4],
                [0, 3, 0, 0, 6, 0, 0, 0, 7],
                [0, 0, 9, 1, 0, 0, 2, 8, 0],
                [0, 0, 6, 0, 2, 8, 0, 0, 0]]

    # answer = recuit_sim(examp)
    # print "Victory!"+str(answer)
    # cost = calc_cost(ex2)
    # print cost

    original = copy.deepcopy(sudogrid)

    lenght_chain = 0

    # creates the matrix that separates fixed vs non-fixed value
    fixed = [[0 for x in range(len(sudogrid))] for x in range(len(sudogrid))]
    ls = len(sudogrid)
    magic_num = math.sqrt(ls)
    for x in range(len(sudogrid)):
        for y in range(len(sudogrid)):
            if sudogrid[x][y] == 0:
                fixed[x][y] = 'n'
                lenght_chain += 1
            else:
                fixed[x][y] = 'f'
    lenght_chain **= 2
    t0 = 4.25  # 4.35  # calc_std_dev(sudogrid, fixed, magic_num)
    delta = 0.985
    print 'delta: ' + str(delta)
    # print 'T0: ' + str(t0)
    t = t0
    max_markov_chains = 40
    print 'mmc: ' + str(max_markov_chains)

    found = False
    sudo = None
    failedwith2count = 0
    bestsofar = sys.maxint
    oldoldcost = sys.maxint
    cost2list = []
    while found is False:

        print "cost2list:" + str(len(cost2list))
        print str(cost2list)
        # print 'RESTART'
        # copyx = copy.deepcopy(sudogrid)
        if oldoldcost == 2:
            failedwith2count += 1
            print "failed with count a 2: " + str(failedwith2count)
        if oldoldcost < bestsofar:
            bestsofar = oldoldcost
        sudo = assign_random_num(sudogrid, fixed, magic_num)
        print "random: " + str(sudo)
        # t0 = calc_std_dev(sudo, fixed, magic_num)
        # cost_dico_main = build_cost_dico(sudo)
        cost_list_main = build_cost_list(sudo)
        # oldoldcost = sum(cost_dico_main.values())
        # total = 0
        # for lis in cost_list_main:
        #     total += sum(lis)
        oldoldcost = calc_total_cost(sudo)
        t = reheat(t0)

        it_no_better = 0
        while it_no_better < max_markov_chains:
            print 'best so far: ' + str(bestsofar)
            print failedwith2count
            print 'actual cost: ' + str(oldoldcost)
            print 'temp: ' + str(t)

            # cost_dico = cost_dico_main
            cost_list = cost_list_main
            oldcost = oldoldcost
            localsudo = copy.deepcopy(sudo)
            for y in range(lenght_chain):
                cost1 = oldcost
                # costbestsudo = calc_cost(sudo)
                # backupsudo = copy.deepcopy(sudo)
                tuplex = swap(localsudo, fixed, magic_num)
                newsudo = tuplex[0]

                changed_rows = list(set(tuplex[1]))
                changed_cols = list(set(tuplex[2]))
                var = 0
                # tempdict = copy.deepcopy(cost_dico)
                templist = copy.deepcopy(cost_list)
                for row in changed_rows:
                    # oldval = cost_dico['row'+str(row)]
                    # newval = calc_cost_row(newsudo, row)
                    # var += (newval-oldval)
                    # tempdict['row'+str(row)] = newval
                    oldval = cost_list[0][row]
                    newval = calc_cost_row(newsudo, row)
                    var += (newval-oldval)
                    templist[0][row] = newval

                for col in changed_cols:
                    oldval = cost_list[1][col]
                    newval = calc_cost_col(newsudo, col)
                    var += (newval-oldval)
                    templist[1][col] = newval
                    # oldval2 = cost_dico['col'+str(col)]
                    # newval2 = calc_cost_col(newsudo, col)
                    # var += (newval2-oldval2)
                    # tempdict['col'+str(col)] = newval2

                cost2 = cost1 + var
                # cost2 = calc_total_cost(newsudo)
                # cost2 = calc_cost(newsudo)
                if cost2 == 0:
                    print '!!!!' + str(newsudo)
                    localsudo = newsudo
                    oldcost = cost2
                    cost_list = templist
                    # cost_dico = tempdict
                    break

                # if cost2 == 2:
                #     print 2
                    # pdb.set_trace()

                if cost2 < cost1:
                    localsudo = newsudo
                    if cost2 == 2:
                        if localsudo not in cost2list:
                            cost2list.append(localsudo)
                        #print str(datetime.datetime.now()) + str(localsudo)
                    oldcost = cost2
                    cost_list = templist
                    # cost_dico = tempdict

                elif cost2 >= cost1:
                    deltacost = cost1 - cost2
                    proba = math.exp(deltacost / t)
                    r = random.random()
                    if r <= proba:
                        localsudo = newsudo
                        oldcost = cost2
                        cost_list = templist
                        # cost_dico = tempdict
                else:
                    pass
                # if sum(cost_dico.values()) != sum(build_cost_dico(localsudo).values()):
                #     print 'fuck'
                #     print y
            # newcost = calc_cost(localsudo)
            if oldcost == 0:
                found = True
                sudo = localsudo
                break
            if oldcost >= oldoldcost:
                # sudo = localsudo  # test!!!!
                it_no_better += 1
            else:
                oldoldcost = oldcost
                sudo = localsudo
                # cost_dico_main = cost_dico
                cost_list_main = cost_list
                it_no_better = 0

            t = cool(t, delta)

    print "Victory!: " + str(sudo)


def root_solution(n):
    x = 0
    grid = [[0 for x in range(n ** 2)] for x in range(n ** 2)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, (n ** 2) + 1):
                w = (n * (i - 1) + j) - 1
                y = k - 1
                grid[w][y] = x % n ** 2 + 1
                x += 1
            x += n
        x += 1

    print str(grid)


def assign_random_num(sudo, fixedgrid, magic_num):
    sudogrid = copy.deepcopy(sudo)
    # first let's get a list of already present number in each squares
    present_number = {}
    for i, v in enumerate(range(0, len(sudogrid), int(magic_num))):
        for j, w in enumerate(range(0, len(sudogrid), int(magic_num))):
            present_num_in_square = []
            for x in range(int(magic_num)):
                for y in range(int(magic_num)):
                    # here, (i,j) rep the squares and (v+x, w+y) the cell
                    # print (v + x, w + y)
                    # print 'square' + str((i, j))
                    if fixedgrid[v + x][w + y] == 'f':
                        present_num_in_square.append(sudogrid[v + x][w + y])
            present_number[str((i, j))] = present_num_in_square
    # print 'fixednum' + str(present_number)

    for i, v in enumerate(range(0, len(sudogrid), int(magic_num))):
        for j, w in enumerate(range(0, len(sudogrid), int(magic_num))):
            for x in range(int(magic_num)):
                for y in range(int(magic_num)):
                    # here, (i,j) rep the squares and (v+x, w+y) the cell
                    if fixedgrid[v + x][w + y] == 'n':
                        while True:
                            z = random.randrange(1, 10)
                            if z not in present_number[str((i, j))]:
                                sudogrid[v + x][w + y] = z
                                present_number[str((i, j))].append(z)
                                break

    # print str(sudogrid)
    return sudogrid


def swap(sudogrid, fixedgrid, magic_num):
    sudo = copy.deepcopy(sudogrid)
    affectedrows = None
    affectedcols = None
    i = None
    j = None
    while True:
        i = random.randrange(0, len(sudo))
        j = random.randrange(0, len(sudo))
        if fixedgrid[i][j] == 'n':
            break

    isquare = math.floor(i / magic_num)
    jsquare = math.floor(j / magic_num)

    # done = False
    while True:
        k = random.randrange(0, magic_num)
        k += isquare * magic_num
        k = int(k)
        #if math.floor(k / magic_num) == isquare:
        #while True:  # pas tant random...
        l = random.randrange(0, magic_num)
        l += jsquare * magic_num
        l = int(l)
        if sudo[i][j] != sudo[k][l] and fixedgrid[k][l] == 'n':
            swaptokl = sudo[i][j]
            swaptoij = sudo[k][l]
            sudo[i][j] = swaptoij
            sudo[k][l] = swaptokl
            affectedrows = (i, k)
            affectedcols = (j, l)
            # done = True
            break
        # if done is True:
        #     break

    return sudo, affectedrows, affectedcols


def calc_total_cost(sudogrid):
    cost = 0
    for x in range(len(sudogrid)):
        cost += calc_cost_row(sudogrid, x)

    for y in range(len(sudogrid)):
        cost += calc_cost_col(sudogrid, y)

    return cost


def calc_cost_row(sudogrid, row):
    cost = 0
    for x in range(1, len(sudogrid) + 1):
            if x not in sudogrid[row]:
                cost += 1
    return cost


def calc_cost_col(sudogrid, col):
    cost = 0
    column = []

    for row in sudogrid:
        column.append(row[col])

    for x in range(1, len(sudogrid) + 1):
            if x not in column:
                cost += 1

    return cost


def build_cost_dico(sudogrid):
    dico = {}
    for x in range(len(sudogrid)):
        dico["row"+str(x)] = calc_cost_row(sudogrid, x)

    for y in range(len(sudogrid)):
        dico["col"+str(y)] = calc_cost_col(sudogrid, y)

    return dico


def build_cost_list(sudogrid):

    listrows = [calc_cost_row(sudogrid, x) for x in range(len(sudogrid))]
    listcols = [calc_cost_col(sudogrid, x) for x in range(len(sudogrid))]

    listx = []
    listx.append(listrows)
    listx.append(listcols)

    return listx


def cool(t, delta):
    # t /= 1 + ((math.log(1 + delta)) / 811) * t
    tx = t * delta
    return tx


def coolv2(t, delta):
    t /= 1 + ((math.log(1 + delta)) / 811) * t
    return t

def coolv3(t, delta, t0):
    t /= 1 + ((math.log(1 + delta)) / (t0+1)) * t
    return t


def calc_std_dev(sudogrid, fixedgrid, magic):
    sudo = copy.deepcopy(sudogrid)
    # sudo = assign_random_num(sudogrid, fixedgrid, magic)
    list_costs = []
    for i in range(1000):
        sudo = swap(sudo, fixedgrid, magic)[0]
        cost = calc_total_cost(sudo)
        list_costs.append(cost)

    x = 0
    for item in list_costs:
        x += item

    mean = x / len(list_costs)

    list_dev = [(x - mean) ** 2 for x in list_costs]

    y = 0
    for item in list_dev:
        y += item

    variance = y / len(list_dev)

    return math.sqrt(variance)


def reheat(t0):
    return t0


if __name__ == "__main__":
    main()