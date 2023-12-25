import torch
import math

class ObservationManager:
    def __init__(self, schemes, default_scheme_name=None, fixed_mask_indices=None):
        self.schemes = schemes
        self.fixed_mask_indices = fixed_mask_indices
        self.random_episode_indices = None
        self.current_scheme = None

        if default_scheme_name:
            self.set_scheme(default_scheme_name)

    def set_scheme(self, scheme_name):
        if scheme_name not in self.schemes:
            raise ValueError(f"Invalid scheme_name: {scheme_name}")

        self.current_scheme = self.schemes[scheme_name]

    def add_scheme(self, new_schemes):
        self.schemes.update(new_schemes)

    def get_observation(self, full_observation):
        if not self.current_scheme:
            raise RuntimeError("No current_scheme is set. Use set_scheme() to set the current scheme.")

        observable_type = self.current_scheme['observable_type']

        if observable_type == 'full':
            return full_observation, torch.tensor([]).long(), full_observation

        D = full_observation.shape[0]
        mask_ratio = self.current_scheme.get('mask_ratio', 0)

        if observable_type == 'fixed':
            mask_indices = self.current_scheme['mask_indices']
        elif observable_type == 'random_step':
            num_masked_elements = math.ceil(D * mask_ratio)
            mask_indices = torch.randperm(D)[:num_masked_elements]
        elif observable_type == 'random_episode':
            if self.random_episode_indices is None:
                num_masked_elements = math.ceil(D * mask_ratio)
                self.random_episode_indices = torch.randperm(D)[:num_masked_elements]
            mask_indices = self.random_episode_indices
        else:
            raise ValueError(f"Invalid observable_type: {observable_type}")

        masked_observation = self.mask_and_add_noise(full_observation, mask_indices)
        return full_observation, mask_indices, masked_observation

    def mask_and_add_noise(self, observation, mask_indices):
        masked_observation = observation.clone()

        if self.current_scheme.get('mask_fill_type', 'zero') == 'zero':
            masked_observation[mask_indices] = 0
        elif self.current_scheme.get('mask_fill_type', 'zero') == 'noise':
            noise_scale = self.current_scheme.get('noise_scale', 0)
            noise = torch.randn(mask_indices.shape) * noise_scale
            masked_observation[mask_indices] +=  noise
        else:
            raise ValueError(f"Invalid mask_fill_type: {self.current_scheme.get('mask_fill_type', 'zero')}")

        return masked_observation

    def reset(self):
        if self.current_scheme and self.current_scheme['observable_type'] == 'random_episode':
            self.random_episode_indices = None
