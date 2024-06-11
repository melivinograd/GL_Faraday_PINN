import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

class GinzburgLandauIntegrator:
    def __init__(self, N, dt, params=None):
        self.N = N
        self.dt = dt
        self.params = params
        self.k = np.arange(0, N/2+1)
        self.rout = None
        self.iout = None

    def initialize_fields(self, x):
        self.rout = np.zeros((self.N, int(steps)))
        self.iout = np.zeros((self.N, int(steps)))
        self.rout[:, 0] = np.real(np.exp(1j * 4 * x)) * 0.1
        self.iout[:, 0] = np.imag(np.exp(1j * 2 * x)) * 0.1

    def compute_abs(self, rf, imf):
        ru = np.fft.irfft(rf)
        iu = np.fft.irfft(imf)
        usq = ru**2 + iu**2
        fsq = np.fft.rfft(usq)
        fsq[int(self.N/3):] = 0
        usq = np.fft.irfft(fsq)
        rout = usq * ru
        iout = usq * iu
        rout = np.fft.rfft(rout)
        iout = np.fft.rfft(iout)
        return rout, iout

    def evolve(self, ru, iu):
        rf = np.fft.rfft(ru)
        imf = np.fft.rfft(iu)
        rft = rf.copy()
        ift = imf.copy()

        for ord in [2, 1]:  # Runge-Kutta loop
            D = self.dt / ord
            rfnl, ifnl = self.compute_abs(rf, imf)
            rfxx = -(self.k**2) * rf
            ifxx = -(self.k**2) * imf
            rfx = 1j * self.k * rf
            ifx = 1j * self.k * imf
            rf = rft + D * (self.params[0] * rf - self.params[1] * rfx +
                            self.params[2] * (rfxx - self.params[3] * ifxx) -
                            self.params[4] * (rfnl - self.params[5] * rfnl))
            imf = ift + D * (self.params[0] * imf - self.params[1] * ifx +
                             self.params[2] * (ifxx + self.params[3] * rfxx) -
                             self.params[4] * (ifnl + self.params[5] * ifnl))
            rf[int(self.N/3):] = 0  # Dealiasing
            imf[int(self.N/3):] = 0  # Dealiasing

        rout = np.fft.irfft(rf)
        iout = np.fft.irfft(imf)
        return rout, iout

    def integrate(self, steps):
        for i in range(int(steps) - 1):  # Temporal evolution
            self.rout[:, i + 1], self.iout[:, i + 1] = self.evolve(self.rout[:, i], self.iout[:, i])
        return self.rout, self.iout

    def plot_field_intensity(self):
        field_intensity = self.rout**2 + self.iout**2
        plt.imshow(field_intensity.T, aspect='auto')
        plt.colorbar()
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Time Step')
        plt.show()

    def plot_time_steps(self):
        from matplotlib.cm import get_cmap
        time_steps = [0, int(steps/4), int(steps/2), int(3*steps/4), int(steps)-1]
        cmap = get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(time_steps)))
        plt.figure(figsize=(10, 6))
        for i, (t,color) in enumerate(zip(time_steps, colors)):
            field_intensity = self.rout[:, t]**2 + self.iout[:, t]**2
            plt.plot(field_intensity + i * 0.05, label=f'Time step = {t}',color=color)
        plt.xlabel('Spatial Coordinate')
        plt.ylabel('Field Intensity (offset)')
        plt.show()

    def save_dataset(self, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def generate_dataset(self, param_ranges, num_samples, steps, save_path):
        x = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
        training_data = []
        testing_data = []

        for _ in range(num_samples):
            # Randomly sample parameters within the given ranges
            mu = np.random.uniform(*param_ranges['mu'])
            vg = np.random.uniform(*param_ranges['vg'])
            xisq = np.random.uniform(*param_ranges['xisq'])
            c1 = np.random.uniform(*param_ranges['c1'])
            lrsq = np.random.uniform(*param_ranges['lrsq'])
            c2 = np.random.uniform(*param_ranges['c2'])
            self.params = (mu, vg, xisq, c1, lrsq, c2)

            self.initialize_fields(x)
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
    dt = 1e-5
    steps = int(0.3 // dt)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Spatial coordinate in [0, 2*pi)

    mu = 0.5
    vg = 8
    xisq = 0.8
    c1 = 2
    lrsq = 0.8
    c2 = 1
    params = (mu, vg, xisq, c1, lrsq, c2)

    integrator = GinzburgLandauIntegrator(N, dt, params)
    integrator.initialize_fields(x)
    rout, iout = integrator.integrate(steps)

    ## Plot
    integrator.plot_field_intensity()
    integrator.plot_time_steps()

    # Create dataset
    num_samples = 2
    save_path = './datasets'

    param_ranges = {
        'mu': (0.1, 1.0),
        'vg': (5, 10),
        'xisq': (0.5, 1.0),
        'c1': (1, 3),
        'lrsq': (0.5, 1.0),
        'c2': (0.5, 1.5)
    }

    integrator = GinzburgLandauIntegrator(N, dt)
    integrator.generate_dataset(param_ranges, num_samples, steps, save_path)
