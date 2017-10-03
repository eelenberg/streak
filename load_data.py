import numpy as np

def loadFromLibsvm(fname,nrows,p):
    ''' 
    adapted from http://stackoverflow.com/questions/23872567/how-to-load-dataset-in-libsvm-python
    usage:
    x,y=loadFromLibsvm('rcv1_train.binary',20)
    x,y=loadFromLibsvm('rcv1_train.binary',2300)
    '''
    Matrix = np.zeros((nrows,p))
    target = np.zeros(nrows)
    with open(fname) as f:
        rownum = 1
        for line in f:
            data = line.split()
            target[rownum-1] = float(data[0]) # target value
            # row = []
            for i, (idx, value) in enumerate([item.split(':') for item in data[1:]]):
                # n = int(idx) - (i + 1) #num missing
                # for tmp in range(n):
                #     row.append(0) #for missing
                # row.append(float(value))
                Matrix[rownum-1,int(idx)-1] = np.array(float(value))
            if rownum >= nrows:
                break
            else:
                rownum += 1
    # return data matrix and target values
    return (np.array(Matrix),np.array(target))


def loadFromGreenhouse(fname,rowInds,p):
    Matrix = np.zeros((len(rowInds),p))
    target = np.zeros(len(rowInds))
    for i,rind in enumerate(rowInds):
        s = "%s%04d.dat" % (fname,rind)
        tmp = np.genfromtxt(s, delimiter=' ', skip_footer=1)
        Matrix[i,:] = tmp[:-1,:].T.reshape(1,-1)
        target[i,:] = tmp[-1,:].T.reshape(1,-1)
    return (np.array(Matrix),np.array(target))


def loadFromElectricity(fname,rowInds,p):
    Matrix = np.zeros((len(rowInds),p))
    target = np.zeros(len(rowInds))
    tmp = pd.read_csv(fname,delimiter=';',header=0,index_col=0,decimal=',',nrows=p+1)
    Matrix = tmp.ix[:-1,rowInds].as_matrix().T
    target = tmp.ix[-1,rowInds].as_matrix()
    return (np.array(Matrix),np.array(target))

