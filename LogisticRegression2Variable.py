import numpy as np
import matplotlib.pyplot as plt


plt.style.use('seaborn')

mean_01 = np.array([1,0.5])
cov_01 = np.array([[1,0.1],[0.1,1.2]])

mean_02 = np.array([4,5])
cov_02 = np.array([[1.21,0.1],[0.1,1.3]])


# Normal Distribution
dist_01 = np.random.multivariate_normal(mean_01,cov_01,500)
dist_02 = np.random.multivariate_normal(mean_02,cov_02,500)
# print(dist_01.shape)
# print(dist_02.shape)


plt.figure(0)
plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

data = np.zeros((1000,3))
# print(data.shape)
data[:500,:2] = dist_01
data[500:,:2] = dist_02
data[500:,-1] = 1.0

np.random.shuffle(data)
# print(data[:10])

split = int(0.8*data.shape[0])

x_train = data[:split,:-1]
x_test = data[split:,:-1]

y_train = data[:split,-1]
y_test  = data[split:,-1]

# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def hypo(x, w, b):
    h = np.dot(x, w) + b

    return sigmoid(h)


def error(y_true, x, w, b):
    m = x.shape[0]
    err = 0.0

    for ix in range(m):
        hx = hypo(x[ix], w, b)
        err += y_true[ix] * np.log2(hx) + (1 - y_true[ix]) * np.log2(1 - hx)

    return -err / m


def get_grad(y_true, x, w, b):
    grad_w = np.zeros(x.shape[1])
    grad_b = 0.0

    m = x.shape[0]
    n = x.shape[1]
    for i in range(m):
        hx = hypo(x[i], w, b)
        grad_b += -1 * (y_true[i] - hx)
        for j in range(n):
            grad_w[j] += -1 * (y_true[i] - hx) * x[i][j]

    grad_w /= m
    grad_b /= m

    return [grad_w, grad_b]


def grad_Descent(x, y_true, w, b, lr):
    err = error(y_true, x, w, b)

    [grad_w, grad_b] = get_grad(y_true, x, w, b)
    b = b - lr * grad_b
    for i in range(w.shape[0]):
        w[i] = w[i] - lr * grad_w[i]

    return err, w, b


def predict(x, w, b):
    conf = hypo(x, w, b)

    if conf < 0.5:
        return 0
    else:
        return 1


def accuracy(x_test, y_test, w, b):
    y_pred = []
    for i in range(y_test.shape[0]):
        pred = predict(x_test[i], w, b)
        y_pred.append(pred)

    return 100 * float((y_pred == y_test).sum()) / y_test.shape[0]


errors = []
acc = []
W = 3 * np.random.random(x_train.shape[1], )
B = 5 * np.random.random()

for ix in range(100):
    e, W, B = grad_Descent(x_train, y_train, W, B, lr=0.5)
    errors.append(e)
    acc.append(accuracy(x_test, y_test, W, B))

plt.figure(0)
plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('x1')
plt.ylabel('x2')

x = np.linspace(-4,8,10)
y = -(W[0]*x + B)/W[1]
plt.plot(x,y,color='k')

plt.legend()
plt.show()

# Error Plot
plt.plot(errors)
plt.show()

# Accuracy Plot
plt.plot(acc)
plt.show()
print(accuracy(x_test, y_test, W, B))
