from matplotlib import pyplot as plt, style
from collections import Counter
from functools import reduce
from PIL import Image
import numpy as np, pandas as pd
import warnings, time, random
import sys, re, os
style.use('ggplot')
warnings.filterwarnings('ignore')


class ImgRec():

    datasetfile = 'images/dataset.txt'
    splitby = '--->'

    def __init__(self, path='images/test.png'):
        if os.path.isfile(path):
            ImgRec.createDatasets()
            ImgRec.recognize(path)
        else:
            sys.stderr.write(path+' does not exist!\n')
            sys.stderr.flush()

    @staticmethod
    def createDatasets():
        if not os.path.isfile(ImgRec.datasetfile):
            dataset = open(ImgRec.datasetfile, 'a')
            numbers = range(0, 10)
            versions = range(1,10)
            print('Creating datasets...')

            for num in numbers:
                for ver in versions:
                    try:
                        filepath = 'images/numbers/'+str(num)+'.'+str(ver)+'.png'
                        img = Image.open(filepath)
                        imgarr = np.array(img)
                        imglist = str(imgarr.tolist())
                        savefmt = str(num)+ImgRec.splitby+imglist+'\n'
                        dataset.write(savefmt)
                    except Exception as e:
                        sys.stderr.write(str(e)+'\n')
                        sys.stderr.flush()
            dataset.close()

    @staticmethod
    def recognize(path):
        matched = []

        dataset = open(ImgRec.datasetfile, 'r').read()
        dataset = dataset.split('\n')

        queryimg = Image.open(path)
        queryarr = np.array(queryimg)
        querylist = str(queryarr.tolist())

        for data in dataset:
            if len(data) > 0:
                splitData = data.split(ImgRec.splitby)
                currnum = str(splitData[0])
                currarr = str(splitData[1])

                queryvals = re.findall(r'\d{1,3}', querylist)
                datavals = re.findall(r'\d{1,3}', currarr)

                if len(queryvals) == len(datavals):
                    i=0
                    while i < len(queryvals):
                        if queryvals[i] == datavals[i]:
                            matched.append(int(currnum))
                        i += 1
        count = Counter(matched)
        nums = list(range(10))
        vals = [count[n] for n in nums]
        avg = reduce(lambda x,y:x+y, vals)/ len(vals)

        fig = plt.figure()
        ax1 = plt.subplot2grid((4,4), (0,0), rowspan=1, colspan=4)
        ax2 = plt.subplot2grid((4,4), (1,0), rowspan=3, colspan=4)

        #plt.title('Image Recognition Result')
        plt.xlabel('Numbers')
        plt.ylabel('Accuraccy')

        ax1.imshow(queryarr, interpolation='nearest')
        ax2.bar(nums, vals, color='c', align='center')
        plt.ylim(int(avg))

        xloc = plt.MaxNLocator(12)
        ax2.xaxis.set_major_locator(xloc)

        #print('Result  ---> ' + str(count))
        #print('Average ---> ' + str(avg))
        print('You most likely drew a', str(count.most_common(1)[0][0]))

        plt.show()


class PenDigitsRecognition:

    def __init__(self):
        train_df = pd.read_csv('optdigits.tra')
        test_df = pd.read_csv('optdigits.tes.txt')

        train_values = train_df.values.tolist()
        test_values = test_df.values.tolist()

        random.shuffle(train_values)
        #random.shuffle(test_values)

        train = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
        test = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

        for t in train_values:
            train[t[-1]].append(t[:-1])

        for t in test_values:
            test[t[-1]].append(t[:-1])
            
        correct = 0
        total = 0

        for k in test:
            for v in test[k]:
                result = PenDigitsRecognition.learningAlgorithm(train, v)
                if result == k:
                    correct += 1
                total += 1
        print('Accuracy:',(correct/total))
    
    @staticmethod
    def learningAlgorithm(data, predict, k=5):
    ##    if len(data) >= 5:
    ##        k = len(data)
    ##        if k % 2 == 0:
    ##            k+=1
    ##        warnings.warn('K is less than the total clusters and has been changed to ' + str(k))
        distances = []
        for g in data:
            for v in data[g]:
                euclidean_dist = np.linalg.norm(np.array(v)-np.array(predict))
                distances.append([euclidean_dist, g])
        rank = [n[1] for n in sorted(distances)[:k]]
        result = Counter(rank).most_common(1)[0][0]
        #print(result)
        return result
        
if __name__ == '__main__':
    img = ImgRec()
##    pdigits = PenDigitsRecognition()
