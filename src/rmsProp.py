import numpy as np

def RMSProp(x,y,learning_rate,beta,epsilon,epochs,batch_size):
    data = np.concatenate((x,y.reshape(-1,1)),axis=1)
    np.random.shuffle(data)
    m,n = data.shape
    x_shuffled = np.concatenate((np.ones((m,1)),data[:,0:n-1]),axis=1)
    y_shuffled = data[:,n-1].reshape(-1,1)
    theta = np.zeros((n,1))
    thetas = []
    v_theta = np.zeros((n,1))
    epoch_loss = []
    loss = []

    for ep in range(epochs):
        for i in range(0,m,batch_size):
            h_x = x_shuffled[i:i+batch_size] @ theta
            error = h_x - y_shuffled[i:i+batch_size]

            j = 1/2 * np.sum(error**2) /len(x_shuffled[i:i+batch_size])

            loss.append(j)

            d_theta = (x_shuffled[i:i+batch_size].T @ error) / len(x_shuffled[i:i+batch_size])

            v_theta = beta * v_theta + (1-beta) * (d_theta**2)
            theta = theta - learning_rate * (1/(np.sqrt(v_theta) + epsilon)) * d_theta
            
            thetas.append(theta)
        epoch_loss.append(j)

        if (np.linalg.norm(d_theta) < 0.001 or abs(epoch_loss[ep]-epoch_loss[ep-1]) < 0.001) and ep != 0:
            break
        
    return theta,thetas,loss,x_shuffled,y_shuffled