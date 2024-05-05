import numpy as np

def Adam(x,y,learning_rate,beta1,beta2,epsilon,epochs,batch_size):
    data = np.concatenate((x,y.reshape(-1,1)),axis=1)
    np.random.shuffle(data)
    m,n = data.shape
    x_shuffled = np.concatenate((np.ones((m,1)),data[:,0:n-1]),axis=1)
    y_shuffled = data[:,n-1].reshape(-1,1)
    theta = np.zeros((n,1))
    thetas = []
    v_theta = np.zeros((n,1))
    m_theta = np.zeros((n,1))
    epoch_loss = []
    loss = []

    iter = 0

    for ep in range(epochs):
        for i in range(0,m,batch_size):
            h_x = x_shuffled[i:i+batch_size] @ theta
            error = h_x - y_shuffled[i:i+batch_size]

            j = 1/2 * np.sum(error**2) /len(x_shuffled[i:i+batch_size])

            loss.append(j)

            d_theta = (x_shuffled[i:i+batch_size].T @ error) / len(x_shuffled[i:i+batch_size])

            m_theta = beta1 * m_theta + (1-beta1) * d_theta
            v_theta = beta2 * v_theta + (1-beta2) * (d_theta**2)

            m_bias_correction = m_theta / (1 - beta1**(iter+1))
            v_bias_correction = v_theta / (1 - beta2**(iter+1))
            
            theta = theta - learning_rate * (1/(np.sqrt(v_bias_correction) + epsilon)) * m_bias_correction

            thetas.append(theta)
            iter += 1
        epoch_loss.append(j)

        # stop condition
        if (np.linalg.norm(d_theta) < 0.001 or abs(epoch_loss[ep]-epoch_loss[ep-1]) < 0.001) and ep != 0:
            break
        
    return theta,thetas,loss,x_shuffled,y_shuffled