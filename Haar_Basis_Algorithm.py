import numpy as np
from np_detector.np_detector import np_detector
from power_estimator.power_estimator import Power_Estimator

class Haar_Basis_Algorithm:
    def __init__(self, Haar_matrix, filter_length, control_level, alpha, G, beta_fa, mu, force_initial_selection=False):
        
        self.x_regressor = np.zeros(filter_length)
        N = self.x_regressor.size

        if N & (N - 1):
            raise ValueError("Regressor length must be a power of 2.")
        
        self.Haar_matrix = Haar_matrix
        self.control_level = control_level
        self.control_basis = self.Haar_matrix.find_indexes_level(control_level)
        self.b = np.zeros_like(self.x_regressor)
        self.G = G
        self.beta_fa = beta_fa
        self.mu = mu
        self.force_initial_selection = force_initial_selection

        # initialize the active basis with V0, V1 and the control level
        self.active_basis = set([0, 1])
        self.active_basis.update(Haar_matrix.find_indexes_level(control_level))

        self.step_size = 1.0/len(self.active_basis)
        
        self.mse_power_estimator = Power_Estimator(self.step_size)

        # level indexed dictionary of power estimators 
        self.power_estimators = {
            level: Power_Estimator(alpha)
            for level in range(self.Haar_matrix.max_level + 1)
        }
        
    def step(self, new_input, new_desired_signal):
        self.x_regressor[1:] = self.x_regressor[:-1]
        self.x_regressor[0] = new_input

        z = self.Haar_matrix.transform_matrix.T @ self.x_regressor

        y = self.b@z

        e = new_desired_signal - y

        self.mse_power_estimator.update(e)

        for level, estimator in self.power_estimators.items():
            first_index = self.Haar_matrix.find_first_index_in_level(level)
            estimator.update(z[first_index])

        for active in self.active_basis:
            level = self.Haar_matrix.get_level(active)
            power = self.power_estimators[level].read_estimate()
            self.b[active] =  self.b[active] + 2*self.step_size*e*z[active]/power
            
        return e
        
    def AI(self):
        self.active_basis = set([0, 1])
        self.active_basis.update(self.control_basis)
    
        level = self.control_level
        power = self.power_estimators[level].read_estimate()
        added = 0
    
        for control in self.control_basis:
            if np_detector(
                self.b[control],
                self.beta_fa,
                self.step_size,
                self.mse_power_estimator.read_estimate(),
                power
            ):
                self.active_basis.update(self.Haar_matrix.parents_of(control))
                self.active_basis.update(self.Haar_matrix.children_of(control))
                added += 1

        # In page 636, Ho & Blunt describe an optional first AI "The first three largest Control coefficients are selected to be nonzero if after the first adaptation interval no Control coefficients exceed the threshold"
        if added == 0 and self.force_selection:
            largest = sorted(
                self.control_basis,
                key=lambda i: abs(self.b[i]),
                reverse=True
            )[:3]
    
            for control in largest:
                self.active_basis.update(self.Haar_matrix.parents_of(control))
                self.active_basis.update(self.Haar_matrix.children_of(control))
    
        self.force_selection = False
    
        self.step_size = self.mu / len(self.active_basis)
        self.mse_power_estimator.update_alpha(self.step_size)
        
    def get_adaptation_interval(self):
        return int(self.G/(4*self.step_size))

    def get_time_coefficients(self):
        return self.Haar_matrix.transform_matrix @ self.b

    def initialize_regressor(self, regressor):
        self.x_regressor = np.array(regressor)

    def reset(self):
        self.b = np.zeros_like(self.x_regressor)
        self.mse_power_estimator.reset()

        # initialize the active basis with V0, V1 and the control level
        self.active_basis = set([0, 1])
        self.active_basis.update(self.Haar_matrix.find_indexes_level(self.control_level))

        self.step_size = self.mu/len(self.active_basis)

        for level in range(self.Haar_matrix.max_level + 1):
            self.power_estimators[level].reset()
            
        self.mse_power_estimator.update_alpha(self.step_size)

        self.force_selection = self.force_initial_selection


    def get_npea(self):
        return len(self.active_basis)