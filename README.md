# FinalProject_MNIST_With_SVM
## Huấn luyện SVM
### Dùng linear kernel (hay nói cách khác là không dùng kernel)
- Thử nghiệm với các giá trị khác nhau của siêu tham số C; với mỗi giá trị C, ghi nhận lại: độ
lỗi trên tập training, độ lỗi trên tập validation, thời gian huấn luyện.<br/>
- Bình luận về kết quả [Gợi ý: Theo lý thuyết thì giá trị C ảnh hưởng như thế nào đến quá
trình học (C quá nhỏ thì sao, C quá lớn thì sao)? Kết quả thí nghiệm có phù hợp với lý
thuyết không?]<br/>
### Dùng Gaussian/RBF kernel
- Thử nghiệm với các giá trị khác nhau của siêu tham số C và γ; với mỗi giá trị C và γ, ghi
nhận lại: độ lỗi trên tập training, độ lỗi trên tập validation, thời gian huấn luyện.<br/>
- Bình luận về kết quả.<br/>
### Chọn hàm dự đoán có độ lỗi nhỏ nhất trên tập validation là hàm dự đoán cuối cùng.


## Explain Code
``` python
import numpy as np
import pickle
import gzip
from sklearn.svm import LinearSVC, SVC
from time import time
from sklearn import metrics


def read_mnist(mnist_file):
    """
    Reads MNIST data.
    
    Parameters
    ----------
    mnist_file : string
        The name of the MNIST file (e.g., 'mnist.pkl.gz').
    
    Returns
    -------
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) : tuple
        train_X : numpy array, shape (N=50000, d=784)
            Input vectors of the training set.
        train_Y: numpy array, shape (N=50000)
            Outputs of the training set.
        val_X : numpy array, shape (N=10000, d=784)
            Input vectors of the validation set.
        val_Y: numpy array, shape (N=10000)
            Outputs of the validation set.
        test_X : numpy array, shape (N=10000, d=784)
            Input vectors of the test set.
        test_Y: numpy array, shape (N=10000)
            Outputs of the test set.
    """
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    
    train_X, train_Y = train_data
    val_X, val_Y = val_data
    test_X, test_Y = test_data    
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y

# get datasets
train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')


# Use SVM with Linear_Kernel

def SVM_Linear_Kernel(C):
    linear_svm = LinearSVC(C=C)
    
    linear_svm_time = time()
    linear_svm.fit(train_X, train_Y)
    linear_svm_time = time() - linear_svm_time
    
    # Validation
    linear_svm_time_for_validation = time()
    linear_svm_score = linear_svm.score(val_X, val_Y)
    linear_svm_time_for_validation = linear_svm_time + time() - linear_svm_time_for_validation
    
    print("Done with Validation in %f s" % linear_svm_time_for_validation)

    print("Accuracy of Validation: %0.3f" % linear_svm_score)
    
    
    # Test
    linear_svm_time_for_test = time()
    linear_svm_score = linear_svm.score(test_X, test_Y)
    linear_svm_time_for_test = linear_svm_time + time() - linear_svm_time_for_test
    
    print("Done with Test in %f s" % linear_svm_time_for_test)

    print("Accuracy of Test: %0.3f" % linear_svm_score)


SVM_Linear_Kernel(1.0)


# Use SVM With GaussianRBF Kernel

def SVM_GaussianRBF_Kernel(C, gamma):
    kernel_svm = SVC(C=C, gamma=gamma)
    
    kernel_svm_time = time()
    kernel_svm.fit(train_X, train_Y)
    kernel_svm_time = time() - kernel_svm_time
    
    # Validation
    kernel_svm_time_for_validation = time()
    kernel_svm_score = kernel_svm.score(val_X, val_Y)
    kernel_svm_time_for_validation = kernel_svm_time + time() - kernel_svm_time_for_validation
    
    print("Done with Validation in %f s" % kernel_svm_time_for_validation)

    print("Accuracy of Validation: %0.3f" % kernel_svm_score)
    

    # Test
    kernel_svm_time_for_test = time()
    kernel_svm_score = kernel_svm.score(test_X, test_Y)
    kernel_svm_time_for_test = kernel_svm_time + time() - kernel_svm_time_for_test
    
    print("Done with Test in %f s" % kernel_svm_time_for_test)

    print("Accuracy of Test: %0.3f" % kernel_svm_score)


SVM_GaussianRBF_Kernel(0.1, 0.9)

# Test
train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')

print('train_X.shape =', train_X.shape)
print('train_Y.shape =', train_Y.shape)
print('val_X.shape   =', val_X.shape)
print('val_Y.shape   =', val_Y.shape)
print('test_X.shape  =', test_X.shape)
print('test_Y.shape  =', test_Y.shape)

print('\ntrain_X: min = %.3f, max = %.3f' %(train_X.min(), train_X.max()))
print('train_Y: min = %d, max = %d' %(train_Y.min(), train_Y.max()))
```

