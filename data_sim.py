import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


class DataSim:
    """
    Class that creates simulated data for my machine learning project.  The
    data being created starts from a one-dimensional, linearly-spaced data
    vector, that can then be noised around that vector like a cloud, rotated in
    3 dimensions, translated from the origin-bisected line segment where it
    starts, and then noised up to many (at least 11) dimensions.

    I would have liked to rotate this vector initially in higher dimensions,
    but was surprised to discover (after doing a lot of work in 3-4 dimensions
    for my MS thesis), that rotating in higher dimensions is computationally
    difficult.  The purpose of this class is to help investigate how a specific
    machine learning technique (Self-organizing Maps, SOM) can discover
    structure.
    """

    def __init__(self, num_points=None, half_width=None, data_1d=None,
                 tube_cloud_noise_std=None, tube_cloud_noise_mean=None,
                 rotate_alpha_degrees=None, rotate_beta_degrees=None,
                 rotate_gamma_degrees=None, translate_x=None, translate_y=None,
                 translate_z=None, num_dimensions=7):
        """
        Construct class according to arguments.  Moderately complicated
        structure after this is to allow defaults at both constructor and
        function level.
        """

        self.data_1d = data_1d
        self.data_3d = None
        self.data_hd = None
        if self.data_1d is None:
            if num_points is None:
                self.create_pure_1d_data()
            else:
                self.create_pure_1d_data(num_points=num_points,
                                         val_half_range=half_width)

        if tube_cloud_noise_mean is None:
            self.radially_cloud_data_around_x_axis()
        else:
            self.radially_cloud_data_around_x_axis(
                std=tube_cloud_noise_std, mean_val=tube_cloud_noise_mean)
        if rotate_alpha_degrees is None:
            self.rotate_zyx_3d()
        else:
            self.rotate_zyx_3d(alpha_angle=rotate_alpha_degrees,
                               beta_angle=rotate_beta_degrees,
                               gamma_angle=rotate_gamma_degrees)
        if translate_x is None:
            self.translate_in_3d()
        else:
            self.translate_in_3d(x=translate_x, y=translate_y, z=translate_z)

    def plot_3d_data(self):
        """Display the 3D data for testing."""

        fig = plt.figure(figsize=plt.figaspect(1)*1.5)
        ax = Axes3D(fig)
        x = self.data_3d[0]
        y = self.data_3d[1]
        z = self.data_3d[2]
        plt.plot(x, y, z, 'o', c='r')

        # Next four lines are a hook to trick matplotlib into plotting all
        # 3 dimensions with a unitary aspect ratio.  Without this trick, it
        # scaled all 3 dimensions differently, wildly differently, which made
        # it difficult to see structure in the data
        maxval = self.data_3d.max()
        minval = self.data_3d.min()
        plt.plot([minval], [minval], [minval], 'o', c='w')
        plt.plot([maxval], [maxval], [maxval], 'o', c='w')
        plt.show()

    def create_pure_1d_data(self, num_points=100, val_half_range=10,
                            even_space=True):
        """
        Creates an array of evenly-spaced numbers for futher creation in the
        processing.  Includes a hook for future addition of jitter along the
        array.

        :param num_points: Total number of points desired.  If number is even,
        one will be added to give same number of points both sides of origin.
        :param val_half_range: data begins bisected by origin, following axis
        both directions a distance of val_half_range
        :param even_space: data evenly spaced along axis, or jittered?
        :return: 1D data array
        """
        num_points = int(num_points)
        if even_space:
            if num_points % 2:
                num_points += 1
            self.data_1d = np.array(list(range(num_points+1)))
            divisor = num_points//2 * 1.0
            self.data_1d = (self.data_1d - divisor) * val_half_range / divisor
            zs = np.zeros(len(self.data_1d))
            self.data_3d = np.array([self.data_1d, zs, zs])
            self.data_hd = self.data_3d
        else:
            raise NotImplementedError(
                    "Uneven spacing will be implemented in the future.")
        return self.data_1d

    def radially_cloud_data_around_x_axis(self, data_1d=None,
                                          std=None, mean_val=None):
        """
        This function takes linear data and makes it a cloud around the length
        of the x-axis.

        :param data_1d: linear data to start with
        :param std: standard deviation for both y and z dimensions
        :param mean_val: bias for both y and z directions
        :return: self.data_3d
        """
        if data_1d is None:
            data_1d = self.data_1d
        x = data_1d
        if mean_val is None:
            val_range = x.max() - x.min()
            mean_val = val_range / 40.0
            std = val_range / 20.0
        y = [random.gauss(mean_val, std) for _ in range(len(data_1d))]
        z = [random.gauss(mean_val, std) for _ in range(len(data_1d))]
        self.data_3d = np.array([x, y, z])
        return self.data_3d

    # noinspection PyPep8Naming
    def rotate_zyx_3d(self, data_3d=None,
                      alpha_angle=45, beta_angle=45, gamma_angle=45):
        """
        This function takes data, either linear data or a linear cloud in 3D,
        and rotates that data in 3D.
        :param data_3d: 3d dataset if you wish to use this function on
        something other than the class member data_3d
        :param alpha_angle: in degrees
        :param beta_angle: in degrees
        :param gamma_angle: in degrees
        :return:
        """
        if data_3d is None:
            data_3d = self.data_3d
        # convert angles to radians
        alpha_rad = alpha_angle * np.pi/180.0
        beta_rad = beta_angle * np.pi/180.0
        gamma_rad = gamma_angle * np.pi/180.0

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
        sin_alpha = np.sin(alpha_rad)
        sin_beta = np.sin(beta_rad)
        sin_gamma = np.sin(gamma_rad)

        c1_c2 = cos_alpha * cos_beta
        c1_s2_s3Mc3_s1 = cos_alpha*sin_beta*sin_gamma - cos_gamma*sin_alpha
        s1_s3Pc1_c3_s2 = sin_alpha*sin_gamma + cos_alpha*cos_gamma*sin_beta
        c2_s1 = cos_beta*sin_alpha
        c1_c3Ps1_s2_s3 = cos_alpha*cos_gamma + sin_alpha*sin_beta*sin_gamma
        c3_s1_s2Mc1_s3 = cos_gamma*sin_alpha*sin_beta - cos_alpha*sin_gamma
        Ms2 = -sin_beta
        c2_s3 = cos_beta*sin_gamma
        c2_c3 = cos_beta*cos_gamma

        # rotation matrix as defined for zyx at
        # https://en.wikipedia.org/wiki/Euler_angles
        rotation_matrix = np.array([[c1_c2, c1_s2_s3Mc3_s1, s1_s3Pc1_c3_s2],
                                   [c2_s1, c1_c3Ps1_s2_s3, c3_s1_s2Mc1_s3],
                                   [Ms2, c2_s3, c2_c3]])

        # apply rotation matrix to data
        data_3d_rot = np.dot(np.transpose(data_3d), rotation_matrix.T)
        self.data_3d = np.transpose(data_3d_rot)

        return self.data_3d

    def translate_in_3d(self, data_3d=None, x=10, y=10, z=10):
        """
        Translate the pseudo-1-dimensional data to a different location in
        3D data space.  I foresee using this after the data vector has been
        rotated
        :param data_3d: Pass a 3D array in if you want to use specific set.
        :param x: Distance to translate (addition) in x, y, and z directions.
        :param y:
        :param z:
        :return:
        """
        if data_3d is None:
            data_3d = self.data_3d
        data_3d[0] += x
        data_3d[1] += y
        data_3d[2] += z
        self.data_3d = data_3d

        return self.data_3d

    def noise_to_high_d(self, data_3d=None, noise_level_large=True,
                        num_of_dimensions=7, data_first=True):
        """
        This combines the 3D data matrix already built with random noise in
        several other dimensions to make a higher-dimensional dataset, with
        useful information theoretically concentrated in that one-dimensional
        sequence we started with.
        :param data_3d: Pass a 3D array in if you want to use specific set.
        :param noise_level_large: Do you want other dimensions to have lower
        or higher values?
        :param num_of_dimensions: Total number of dimension for final data
        :param data_first: Should data_3d be first 3 dimensions of set, or
        randomly scattered?
        :return: self.data_hd
        """
        if data_3d is None:
            data_3d = self.data_3d
        length = data_3d.shape[1]
        num_new_dims = num_of_dimensions - 3
        if noise_level_large:
            mult_const = 10.0
        else:
            mult_const = 1.0
        # Create an array of random elements the necessary size for the new
        # dimensions
        new_dims = mult_const * np.random.randn(num_new_dims, length)
        # Stack the 3D data on top of the new dimensions
        self.data_hd = np.concatenate((data_3d, new_dims), axis=0)
        # Shuffle the dimensions if desired
        if not data_first:
            np.random.shuffle(self.data_hd)

        return self.data_hd

