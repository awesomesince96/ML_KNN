import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter

#function to find most frequent occuring class in k nearest neighbour
def most_common(lst):
    return np.bincount(lst).argmax()
    
def train(k,images_train,images_test, labels_train, labels_test):
    print(k,images_train.shape)
    accuracy = 0
    count = 0
    total_train_data = len(labels_train)
    total_test_data = len(labels_test)
    #educliean distance
    ed = [[0 for i in range(images_train.shape[0])] for j in range(total_test_data)]
    sorted_ed = [[0 for i in range(images_train.shape[0])] for j in range(total_test_data)]
    class_classification_total = [0] * 10
    class_classification_true = [0] * 10
    
    #l2 norm
    for i in range(len(images_test)):
        for j in range(len(images_train)):
            ed[i][j] = np.linalg.norm(images_train[j]-images_test[i])   
        sorted_ed[i] = np.argsort(ed[i])
    
    for i in range(len(images_test)):
        for j in range(len(images_train)):
            ind_value = sorted_ed[i][j]
            sorted_ed[i][j] = labels_train[ind_value]
    
    # comparing with neighbour and finding the nearest one by comparing k members
    for i in range(len(images_test)):
        # choose top 3 value from sorted index
        k_neighbour_set = sorted_ed[i][0:k]
        predicted_index = most_common(k_neighbour_set)
        #predicted_index = Counter(k_neighbour_set).most_common(1)[0][0]
       
        #predicted value based on index'
        predicted_value = predicted_index
        real_label = labels_test[i]    
        if predicted_value == real_label:
            count = count + 1
            #count true per class
            class_classification_true[predicted_value] = class_classification_true[predicted_value] + 1 
        #count all hit and categorize in class
        class_classification_total[predicted_value] = class_classification_total[predicted_value] + 1 
        
    for i in range(10):
        accuracy = ( class_classification_true[i] / class_classification_total[i] ) * 100
        print('Accuracy of Class: ',i,' is ', str(round(accuracy,2)),'%')
        
    #calulating accuracy    
    avg_accuracy = ( count / total_test_data ) * 100
    print('Accuracy: ',avg_accuracy,' %') 
    
    return avg_accuracy
    

def main():
    M = loadmat('MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']  
    #just to make all random sequences on all computers the same.
    np.random.seed(1)
    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]
    inds = np.random.permutation(images_test.shape[0])
    
    
    images_test_data = images_test[9000:9100]
    labels_test_data = labels_test[9000:9100]
    
    c = [200,500,700,900,1000,2000,3000,5000,7000,9000]
    k_set = [1,2,3,5,10]
    k = 3
    b = []
    d = []
    e = []
    
    #section b
    for i in range(len(c)):
        images_train_data = images_train[0:c[i],:]
        labels_train_data = labels_train[0:c[i],:] 
        b.append(train(k,images_train_data,images_test_data, labels_train_data, labels_test_data))
    
    plt.xlabel("No of Training Data")
    plt.ylim(0,100)
    plt.ylabel("Average Accuracy at K = 1")
    plt.plot(c,b,'b-')
    plt.show()
    
    #section c
    for i in range(len(k_set)):
        k = k_set[i]
        images_train_data = images_train[0:3000,:]
        labels_train_data = labels_train[0:3000,:] 
        d.append(train(k,images_train_data,images_test_data, labels_train_data, labels_test_data))
    
    plt.xlabel("No of Kth Neighbour")
    plt.xlim(0,12)
    plt.ylim(0,100)
    plt.plot(k_set[0],d[0],'bo')
    plt.plot(k_set[1],d[1],'ro')
    plt.plot(k_set[2],d[2],'yo')
    plt.plot(k_set[3],d[3],'go')
    plt.plot(k_set[4],d[4],'mo')
    plt.ylabel("Average Accuracy")
    plt.show()
    
    #section e
    for i in range(len(k_set)):
        k = k_set[i]
        images_train_data = images_train[0:1000,:]
        labels_train_data = labels_train[0:1000,:] 
        images_test_data = images_test[2000:3000]
        labels_test_data = labels_test[2000:3000]
        e.append(train(k,images_train_data,images_test_data, labels_train_data, labels_test_data))
    
    print(k_set)
    print(e)
    plt.xlabel("No of Kth Neighbour")
    plt.ylabel("Average Accuracy")
    plt.xlim(0,12)
    plt.ylim(0,100)
    plt.plot(k_set[0],e[0],'bo')
    plt.plot(k_set[1],e[1],'ro')
    plt.plot(k_set[2],e[2],'yo')
    plt.plot(k_set[3],e[3],'go')
    plt.plot(k_set[4],e[4],'mo')
    plt.show()
    print('As per my observation k = 5 would be the best fit')
 
    

main()