# import numpy as np
# from dataclasses import dataclass
# from typing import List

# @dataclass
# class Param:
#     name: str 
#     bounds: List[float] = None
#     good_values: List[float] = None 

# # param = Param('p1')


# class RandomSearch:

#     def __init__(self, sample_size, drop_prob, safe_distance):
#         self.sample_size = sample_size
#         self.drop_prob = drop_prob
#         self.safe_distance = safe_distance
#         self.bad_params = []

#     def _sample_from_bounds(self, bounds):
#         return np.random.uniform(bounds[0], bounds[1], size=self.sample_size)

#     def _sample_from_values(self, values_list):
#         return np.random.choice(values_list, size=self.sample_size)

#     def sample_params(self, params):
#         trial_params = np.zeros((self.sample_size, len(params)))
#         for i, param in enumerate(params):
#             assert param.good_values is not None or param.bounds is not None 
#             if param.good_values:
#                 trial_params[:, i] = self._sample_from_values(param.good_values)
#             else:
#                 trial_params[:, i] = self._sample_from_bounds(param.bounds)   
        
#         safe_params = []
#         for point in trial_params:
#             if all([
#                 d >= self.safe_distance for d in 
#                 (np.linalg.norm(self.bad_params - point, axis=1) / np.linalg.norm(self.bad_params, axis=1)) # to fix: divide by bounds width instead
#             ]):
#                 safe_params.append(point)
#         return safe_params

#     def tune(self, model, params, data, budget, epoch=20):
#         total_tried = 0
#         while total_tried < budget:
#             params_next = self.sample_params(params)
#             total_tried += len(params_next)
#             fun_vals = []
#             for param in params_next:
#                 val = model(param, epoch)
#                 fun_vals.append(val)
#             n_to_keep = int((1 - self.drop_prob) * len(params_next))
#             sorted_ind = np.argpartition(fun_vals, n_to_keep)
#             self.bad_params = np.vstack((self.bad_params, params_next[sorted_ind[n_to_keep:]]))
#             # need to save model somehow

