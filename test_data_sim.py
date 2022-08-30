import unittest
from data_sim import DataSim
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# noinspection PyPep8Naming,PyTypeChecker
class TestDataSim(unittest.TestCase):
    def test_data1d_created_w_default(self):
        print("in test_data1d_created_w_default")
        classUnderTest = DataSim()
        result = classUnderTest.create_pure_1d_data()
        classUnderTest.plot_3d_data()
        self.assertEqual(len(result), 101)

    def test_data1d_created_uneven(self):
        print("in test_data1d_created_w_default")
        classUnderTest = DataSim()
        with self.assertRaises(NotImplementedError):
            classUnderTest.create_pure_1d_data(even_space=False)

    def test_data_clouded_w_default(self):
        print("in test_data_clouded_w_default")
        data_1d = np.array(range(-100, 100)) / 10.0
        classUnderTest = DataSim()
        result = classUnderTest.radially_cloud_data_around_x_axis(
            data_1d=data_1d)
        classUnderTest.plot_3d_data()
        self.assertEqual(result.shape, (3, 200))

    def test_data_rotated_w_default(self):
        print("in test_data_rotated_w_default")
        data_1d = np.array(range(-100, 100))/10.0
        zs = np.zeros(len(data_1d))
        data_3d = np.array([data_1d, zs, zs])
        classUnderTest = DataSim()
        result = classUnderTest.rotate_zyx_3d(data_3d=data_3d)
        classUnderTest.plot_3d_data()
        self.assertEqual(result.shape, (3, 200))

    def test_translate_w_default(self):
        print("in test_translate_w_default")
        data_1d = np.array(range(-100, 100))/10.0
        zs = np.zeros(len(data_1d))
        data_3d = np.array([data_1d, zs, zs])
        classUnderTest = DataSim()
        classUnderTest.rotate_zyx_3d(data_3d=data_3d)
        result = classUnderTest.translate_in_3d()
        classUnderTest.plot_3d_data()
        self.assertEqual(result.shape, (3, 200))

    def test_constructor_defaults(self):
        print("in test_constructor_defaults")
        classUnderTest = DataSim()
        classUnderTest.plot_3d_data()
        self.assertEqual(classUnderTest.data_3d.shape, (3, 101))

    def test_constructor_bigcloud_no_rotate(self):
        print("in test_constructor_bigcloud_no_rotate")
        classUnderTest = DataSim(tube_cloud_noise_mean=4,
                                 tube_cloud_noise_std=2,
                                 rotate_alpha_degrees=0,
                                 rotate_beta_degrees=0,
                                 rotate_gamma_degrees=0)
        classUnderTest.plot_3d_data()

    def test_constructor_bigcloud(self):
        print("in test_constructor_bigcloud")
        classUnderTest = DataSim(tube_cloud_noise_mean=4,
                                 tube_cloud_noise_std=2)
        classUnderTest.plot_3d_data()

    def test_high_d(self):
        print("in test_high_d")
        classUnderTest = DataSim()
        result = classUnderTest.noise_to_high_d(num_of_dimensions=8)
        self.assertEqual(result.shape, (8, 101))

    def test_high_d_shuffle(self):
        print("in test_high_d")
        classUnderTest = DataSim()
        result = classUnderTest.noise_to_high_d(num_of_dimensions=8,
                                                data_first=False)
        self.assertEqual(result.shape, (8, 101))

if __name__ == '__main__':
    unittest.main()
