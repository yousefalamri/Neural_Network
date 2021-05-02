import numpy as np

############################# supporting functions for Neural Network ################################
# Load train.csv and test.csv
with open('bank-note/train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

with open('bank-note/test.csv') as f:
    testing_data = [];
    for line in f:
        terms = line.strip().split(',')
        testing_data.append(terms)


def convert_to_float(input_data):
    for elem in input_data:
        for i in range(len(input_data[0])):
            elem[i] = float(elem[i])
    return input_data


# augment feature vector
def augment_feature_vector(input_data):
    labels = [elem[-1] for elem in input_data]
    data_list = input_data
    for i in range(len(input_data)):
        data_list[i][-1] = 1.0
    for i in range(len(input_data)):
        data_list[i].append(labels[i])
    return data_list


# convert (0,1) labels to (-1,1)
def convert_to_pm_one(input_data):
    new_list = input_data
    for i in range(len(input_data)):
        if new_list[i][-1] != 1.0:
            new_list[i][-1] = -1.0
    return new_list


training_data = convert_to_float(training_data)  # convert to float  types data
testing_data = convert_to_float(testing_data)
train_data = augment_feature_vector(convert_to_pm_one(training_data))
test_data = augment_feature_vector(convert_to_pm_one(testing_data))
N_train = len(train_data)
N_test = len(test_data)
ftr_len = len(train_data[0]) - 1

# sign function
def sgn(input):
    sign = 0
    if input > 0:
        sign = 1
    else:
        sign = -1
    return sign

# counting the number of errors
def count_error(prediction, actual):
    error_count = 0
    input_length = len(prediction)
    for i in range(input_length):
        if prediction[i] != actual[i]:
            error_count = error_count + 1
    return error_count / input_length * 100.0

# sigmoid function
@np.vectorize
def sigmoid(s):
    if s < -100:
        sigma = 0
    else:
        sigma = 1 / (1 + np.e ** (-s))
    return sigma

# rate schedule
def lrn_rate(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)

# Initialize random weights
def InitializeWeights(size):
    mean = 0
    stdev = 1
    return np.random.normal(mean, stdev, size)

# Initialize zero weights
def InitializeWeightsZeros(size):
    return np.zeros(size)


#############################  functions for Neural Network ################################

class ThreeLayersNeuralNetwork:
    def __init__(self, num_inputs, num_outputs, num_lyr_1, num_lyr_2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_lyr_1 = num_lyr_1
        self.num_lyr_2 = num_lyr_2

        # Initialize random weights
        self.w_mat_lyr_0 = InitializeWeights([self.num_lyr_1 - 1, self.num_inputs])
        self.w_mat_lyr_1 = InitializeWeights([self.num_lyr_2 - 1, self.num_lyr_1])
        self.w_mat_lyr_2 = InitializeWeights([self.num_outputs, self.num_lyr_2])

        # # Initialize zero weights
        # self.w_mat_lyr_0 = InitializeWeightsZeros([self.num_lyr_1 - 1, self.num_inputs])
        # self.w_mat_lyr_1 = InitializeWeightsZeros([self.num_lyr_2 - 1, self.num_lyr_1])
        # self.w_mat_lyr_2 = InitializeWeightsZeros([self.num_outputs, self.num_lyr_2])

    # # Uncomment the following to test the pen paper problem
    #     self.w_mat_lyr_0 = np.array([[-2,-3,-1],[2,3,1]])
    #     self.w_mat_lyr_1 = np.array([[-2,-3,-1],[2,3,1]])
    #     self.w_mat_lyr_2 = np.array([[2, -1.5, -1]])


    # This function returns the z-values for layer 1 and 2
    def ForwardPassNN(self, input_vec):
        # Calculate the output of each layer Z and the final output Y
        input_vec = np.array(input_vec, ndmin=2).T
        Z1 = np.dot(self.w_mat_lyr_0, input_vec)
        Z1 = sigmoid(Z1)
        Z2 = np.dot(self.w_mat_lyr_1, np.concatenate((Z1, [[1]]), axis=0))
        Z2 = sigmoid(Z2)
        Y = np.dot(self.w_mat_lyr_2, np.concatenate((Z2, [[1]]), axis=0))

        # # Uncomment the following to test the pen paper problem
        # output_node_1 = np.concatenate((Z1, [[1]]), axis = 0)
        # output_node_2 = np.concatenate((Z2, [[1]]), axis = 0)
        # return [Y, output_node_1, output_node_2]
        return sgn(Y)

    # This function evaluate the gradients at each layer using BP
    def UpdateWeights(self, input_vec, true_label, gamma_0, d):
        # Calculate the output of each layer Z and the final output Y
        input_vec = np.array(input_vec, ndmin=2).T
        Z1 = sigmoid(np.dot(self.w_mat_lyr_0, input_vec))
        Z2 = sigmoid(np.dot(self.w_mat_lyr_1, np.concatenate((Z1, [[1]]), axis=0)))
        Y = np.dot(self.w_mat_lyr_2, np.concatenate((Z2, [[1]]), axis=0))

        # Calculate final prediction error in last layer
        Error_Y = Y - true_label

        # calculate the gradient matrix at layer 2
        dLdW2 = Error_Y * (np.concatenate((Z2, [[1]]), axis=0)).T

        # calculate the error/gradient matrix at layer 1
        Error_lyr_2 = Error_Y * (self.w_mat_lyr_2[0,:][:-1]) * (Z2.T) * (1 - (Z2.T))
        dLdW1 = np.dot(np.concatenate((Z1, [[1]]), axis=0) , Error_lyr_2).T

        # calculate the error/gradient matrix at layer 1
        W_times_Z = self.w_mat_lyr_2[0,:][:-1] * Z2.T * (1.0-Z2.T)
        temp_prod_all_nrns = np.zeros((self.num_lyr_1 - 1, 1))
        for n in range(self.num_lyr_1 - 1):
            temp_prod_all_nrns[n, 0] = Error_Y * np.inner(W_times_Z, self.w_mat_lyr_1[:, n].T) * (Z1.T)[0, n] * (1.0 - (Z1.T)[0, n])
        dLdW0 = np.dot(temp_prod_all_nrns, input_vec.T)

        # update weights
        t = 1
        self.w_mat_lyr_2 = self.w_mat_lyr_2 - lrn_rate(t, gamma_0, d) * dLdW2
        self.w_mat_lyr_1 = self.w_mat_lyr_1 - lrn_rate(t, gamma_0, d) * dLdW1
        self.w_mat_lyr_0 = self.w_mat_lyr_0 - lrn_rate(t, gamma_0, d) * dLdW0


########################################### Run Neural Network ################################################
Width = 100
gamma_0 = 0.01
d = 2

# Initialize the structure and weights of the network
MyNetwork = ThreeLayersNeuralNetwork(num_inputs = ftr_len, num_outputs = 1, num_lyr_1 = Width, num_lyr_2 = Width)

# train the neural network
for i in range(N_train):
    MyNetwork.UpdateWeights(train_data[i][0:ftr_len], train_data[i][-1], gamma_0, d)

# Apply neural net on training data
training_prediction = []
for i in range(N_train):
    actual_training_labels = [elem[-1] for elem in train_data]
    training_prediction.append(MyNetwork.ForwardPassNN(train_data[i][0:ftr_len]))
print('training error % = ', count_error(training_prediction, actual_training_labels))

# Apply neural net on testing data
testing_prediction = []
for i in range(N_test):
    actual_testing_labels = [elem[-1] for elem in test_data]
    testing_prediction.append(MyNetwork.ForwardPassNN(test_data[i][0:ftr_len]))
print('testing error % = ', count_error(testing_prediction, actual_testing_labels))

