import numpy as np

class NLMS_Algorithm:
    def __init__(self, filter_length, mu, epsilon=1E-16):
        
        self.x_regressor = np.zeros(filter_length)

        self.w = np.zeros(filter_length)
        
        self.step_size = mu

        self.epsilon = epsilon
        
    def step(self, new_input, new_desired_signal):
        self.x_regressor[1:] = self.x_regressor[:-1]
        self.x_regressor[0] = new_input

        y = self.w@self.x_regressor

        e = new_desired_signal - y

        self.w += self.step_size*e*self.x_regressor/(self.x_regressor@self.x_regressor + self.epsilon)
            
        return e
        
    def AI(self):
        pass   
        
    def get_adaptation_interval(self):
        return 67

    def get_time_coefficients(self):
        return self.w

    def initialize_regressor(self, regressor):
        self.x_regressor = np.array(regressor)

    def reset(self):
        self.w = np.zeros_like(self.x_regressor)

    def get_npea(self):
        return len(self.x_regressor)