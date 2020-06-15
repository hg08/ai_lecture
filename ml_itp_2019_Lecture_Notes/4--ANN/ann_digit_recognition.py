# neural network class definition
class neural_network:
    # initialise the neural network
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        
        # link weight matrices, wih and who
        self.wih = (np.random.rand(self.hnodes, self.inodes)-0.5)
        self.who = (np.random.rand(self.onodes, self.hnodes)-0.5)
        
        #learning rate
        self.lr = learning_rate
        
        # the activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    # train the neural network
    def train(self, inputs_list,targets_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs =self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the final outputs
        final_outputs = self.activation_function(final_inputs)
        
        # error is the (target -actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors,split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0-final_outputs)), np.transpose(hidden_outputs) )
        # update the weights for the links between the input and hidden ayers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)), np.transpose(inputs) )        
        pass
            
        
    # query the neural network
    def query(self,inputs_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list,ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs =self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the final outputs
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def score(inputs):
        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (np.asfarray(all_values[1:])/255.0* 0.99) +0.01
            # query the network
            outputs = n.query(inputs)
            label = np.argmax(outputs) # the mode
            # append correct or incorrect to list
            if (label == correct_label):
                #match correct answer
                scorecard.append(1)
            else:
                #does not match correct answer
                scorecard.append(0)
                pass
            pass
        scorecard_array = np.asarray(scorecard)
        return scorecard_array.sum()/scorecard_array.size


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.special # to use the Sigmoid function in special module
    
    # set the numbers of nodes
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    # load the mnist training data csv file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # load the mnist testing data csv file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv",'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    #train
    lr_list = 0.05*np.arange(10)[1:10]
    sc_list = []
    for learning_rate in lr_list:
        # create a neuralnetwork
        n = neural_network(input_nodes,hidden_nodes,output_nodes,learning_rate) 
        # travel over all samples
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
            #create the target output values
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs,targets)
            pass
        
        sc_list.append(score(inputs))
    
    fig, ax = plt.subplots(1, 1)  # a figure with a 1X1 grid of Axes
    fig.suptitle('The accuracy of the ANN classifier')  # Add a title so we know which it is
    ax.plot(lr_list,sc_list,'-o')
    plt.xlabel("learning rate")
    plt.ylabel("accuracy")
    plt.grid()
    plt.savefig('accuracy_lr.png')
    plt.show()    