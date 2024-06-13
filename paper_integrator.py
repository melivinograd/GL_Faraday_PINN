import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

class GinzburgLandauIntegrator:
    def __init__(self, N, dt, params=None, x_lim=np.pi):
        self.N = N
        self.dt = dt
        self.params = params
        self.k = np.fft.rfftfreq(N, 1.0/N)  # Use rfftfreq for real FFT
        self.x = np.linspace(-x_lim, x_lim, N, endpoint=False)
        self.gamma_x = params[3] * np.exp(-self.x**2 / (2 * params[4]**2))
        self.gamma_x = params[3] * np.ones(self.N)
        self.gamma_0 = params[3]
        self.rout = None
        self.iout = None


    def initialize_fields(self, noise_amplitude=0.0, shape='periodic'):
        self.rout = np.zeros((self.N, int(steps)))
        self.iout = np.zeros((self.N, int(steps)))

        if shape == "uniform":
            base_r = np.ones(self.N) * 0.1
            base_i = np.zeros(self.N)

            # Adding noise
            noise_r = noise_amplitude * np.random.randn(self.N)
            noise_i = noise_amplitude * np.random.randn(self.N)

            self.rout[:, 0] = base_r + noise_r
            self.iout[:, 0] = base_i + noise_i

        elif shape == 'random':
            # Random initial condition
            self.rout[:, 0] = noise_amplitude * np.random.randn(self.N)
            self.iout[:, 0] = noise_amplitude * np.random.randn(self.N)

        elif shape == 'localized':
            # Localized perturbation
            base_r = np.zeros(self.N)
            base_i = np.zeros(self.N)
            center = self.N // 2
            width = self.N // 20
            base_r[center-width:center+width] = 1.0

            # Adding noise
            noise_r = noise_amplitude * np.random.randn(self.N)
            noise_i = noise_amplitude * np.random.randn(self.N)

            self.rout[:, 0] = base_r + noise_r
            self.iout[:, 0] = base_i + noise_i

        elif shape == 'periodic':
            # Periodic initial condition
            self.rout[:, 0] = np.real(np.exp(1j * 4 * self.x)) * 0.1
            self.iout[:, 0] = np.imag(np.exp(1j * 2 * self.x)) * 0.1

    def evolve(self, ru, iu):
        rf = np.fft.rfft(ru)
        imf = np.fft.rfft(iu)
        rft = rf.copy()
        ift = imf.copy()

        for ord in [2, 1]:  # Runge-Kutta loop
            D = self.dt / ord

            rfxx = -(self.k**2) * rf
            ifxx = -(self.k**2) * imf
            abs_A2 = np.fft.irfft(rf)**2 + np.fft.irfft(imf)**2
            abs_A2_fft = np.fft.rfft(abs_A2)

            gamma_A_star_real = np.fft.rfft(self.gamma_x * np.fft.irfft(rf))
            gamma_A_star_imag = np.fft.rfft(self.gamma_x * np.fft.irfft(imf))

            rf = rft + D * (-self.params[0] * rf + self.params[1] * imf + self.params[2] * ifxx + rf * abs_A2_fft + gamma_A_star_real)
            imf = ift + D * (-self.params[0] * imf - self.params[1] * rf - self.params[2] * rfxx - imf * abs_A2_fft - gamma_A_star_imag)

            rf[int(self.N / 3):] = 0  # Dealiasing
            imf[int(self.N / 3):] = 0  # Dealiasing

        rout = np.fft.irfft(rf, n=self.N)
        iout = np.fft.irfft(imf, n=self.N)
        return rout, iout

    def integrate(self, steps):
        for i in range(int(steps) - 1):  # Temporal evolution
            print(f'Step={i}')
            self.rout[:, i + 1], self.iout[:, i + 1] = self.evolve(self.rout[:, i], self.iout[:, i])
        return self.rout, self.iout

    def plot_field_intensity(self):
        field_intensity = self.rout**2 + self.iout**2
        plt.imshow(field_intensity.T, aspect='auto')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('Step')
        plt.title(fr'$\gamma_0=${self.gamma_0}')
        plt.show()

    def plot_time_steps(self):
        from matplotlib.cm import get_cmap
        time_steps = [0, int(steps/4), int(steps/2), int(3*steps/4), int(steps)-1]
        cmap = get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(time_steps)))
        plt.figure(figsize=(10, 6))
        for i, (t, color) in enumerate(zip(time_steps, colors)):
            field_intensity = self.rout[:, t]**2 + self.iout[:, t]**2
            plt.plot(field_intensity + i * 0.05, label=f'Time step = {t}', color=color)
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Field Intensity (offset)')
        plt.show()

    def save_dataset(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def generate_dataset(self, param_ranges, num_samples, steps, save_path):
        training_data = []
        testing_data = []

        for _ in range(num_samples):
            # Randomly sample parameters within the given ranges
            mu = np.random.uniform(*param_ranges['mu'])
            nu = np.random.uniform(*param_ranges['nu'])
            alpha = np.random.uniform(*param_ranges['alpha'])
            gamma0 = np.random.uniform(*param_ranges['gamma'])
            sigma = np.random.uniform(*param_ranges['sigma'])
            self.params = (mu, nu, alpha, gamma0, sigma)
            self.gamma_x = gamma0 * np.exp(-self.x**2 / (2 * sigma**2))

            self.initialize_fields()
            rout, iout = self.integrate(steps)

            # Split data into training and testing
            if np.random.rand() < 0.8:  # 80% training, 20% testing
                training_data.append((self.params, rout, iout))
            else:
                testing_data.append((self.params, rout, iout))

        os.makedirs(save_path, exist_ok=True)
        self.save_dataset(training_data, os.path.join(save_path, 'training_data.pkl'))
        self.save_dataset(testing_data, os.path.join(save_path, 'testing_data.pkl'))

if __name__ == "__main__":
    # Try one
    N = 256
    dt = 1e-5  # Smaller time step
    steps = int(6e3 // dt)
    #  steps = 10000

    mu = 0.45
    nu = 1.0
    alpha = 1.0
    gamma0 = 0.90
    sigma = 16.0
    params = (mu, nu, alpha, gamma0, sigma)

    integrator = GinzburgLandauIntegrator(N, dt, params, x_lim=8)
    integrator.initialize_fields()
    rout, iout = integrator.integrate(steps)

    # Plot
    integrator.plot_field_intensity()
    integrator.plot_time_steps()

    #  Create dataset
    #  num_samples = 2
    #  save_path = './datasets'

    #  param_ranges = {
        #  'mu': (0.1, 1.0),
        #  'nu': (0.0, 0.5),
        #  'alpha': (0.5, 2.0),
        #  'gamma': (0.1, 0.5),
        #  'sigma': (0.5, 2.0)
    #  }

    #  integrator = GinzburgLandauIntegrator(N, dt)
    #  integrator.generate_dataset(param_ranges, num_samples, steps, save_path)

