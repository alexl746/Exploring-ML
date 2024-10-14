#Manual basic machine training without pytorch
#training a machine to learn what f(x) is  (5x)
import numpy as np

uinput =input("What multiplier would you like f(x) to be of x? (Input only integers)")
uinputi = int(uinput)
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([uinputi,2*uinputi,3*uinputi,4*uinputi], dtype=np.float32)

w = 0.0

#model prediction
def forward(x):
    return w * x

#loss = mean squared error
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

#gradient
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

print(f"prediction before training: f(5): {forward(5):.3f}")

#training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    l = loss(Y,y_pred)

    dw = gradient(X,Y,y_pred)
    
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}:w = {w:.3f}, loss = {l:.8f}")

print(f"prediction after training: f(5) = {forward(5):.3f}")
print(f"prediction after training: f(10) = {forward(10):.3f}")
