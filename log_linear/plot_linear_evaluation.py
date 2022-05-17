from fileinput import filename
import matplotlib.pyplot as plt
import os
def plot_linear_evaluation(filename = 'log_MAE_2_Linear.txt'):
    with open(filename,'r') as f:
        L = f.readlines()
        L = L[6:]
    id = 0
    X = []
    Y = []
    for l in L:
        if id%2==0:
            l = l.replace('\n','')
            s = l.split(' ')
            if '' in s:
                s.remove('')
            Y.append(float(s[4]))
            ss = s[3].split(',')
            X.append(int(ss[0]))
        id += 1
    BestAccuracy = max(Y)
    plt.figure()
    plt.title(filename[:-4]+" best accuracy:%s"%BestAccuracy)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(X, Y, label=filename.split('_')[1]+filename.split('_')[2],  color="#F9A602", marker='*', linestyle="-")
    plt.legend()
    plt.savefig("Linear Evaluation "+filename.split('_')[1]+filename.split('_')[2]+'.png')
    plt.show()
if __name__ == '__main__':
    filenames = os.listdir()
    for filename in filenames:
        if '.txt' in filename:
            plot_linear_evaluation(filename=filename)
    print('visualization of linear evaluation completed!')
    