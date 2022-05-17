from fileinput import filename
import matplotlib.pyplot as plt
import os
from random import random
def plot_linear_evaluation(filename = 'log_MAE_1_New.txt'):
    with open(filename,'r') as f:
        L = f.readlines()
    X = []
    Y = []
    for l in L:
        if '- epoch' in l:
            l = l.replace('\n','')
            s = l.split(' ')
            if '' in s:
                s.remove('')
            # print(s)
            Y.append(float(s[6]))
            # print(Y)
            ss = s[2].split(',')
            X.append(int(ss[0]))
            # assert 1==0
    plt.figure()
    plt.title("Training loss curve:"+filename[:-8])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(X, Y, label=filename.split('_')[1]+filename.split('_')[2],  color="#8552A1", marker='*', linestyle="-")
    plt.legend()
    plt.savefig("Training loss curve:"+filename.split('_')[1]+filename.split('_')[2]+'.png')
    plt.show()
    label = filename.split('_')[1]+filename.split('_')[2]
    color = (random(),random(),random())
    return X,Y,label,color

if __name__ == '__main__':
    filenames = sorted(os.listdir())
    Xs = []
    Ys = []
    Ls = []
    Cs = []
    for filename in filenames:
        if '.txt' in filename:
            X,Y,label,color = plot_linear_evaluation(filename=filename)
            Xs.append(X)
            Ys.append(Y)
            Ls.append(label)
            Cs.append(color)
    l = len(Xs)
    plt.figure()
    plt.title("Training loss curve all")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    for i in range(l):
        plt.plot(Xs[i], Ys[i], label=Ls[i],  color=Cs[i], marker='*', linestyle="-")
    plt.legend()
    plt.savefig("Training loss curve all.png")
    plt.show()
    print('visualization of pretraing loss completed!')
    