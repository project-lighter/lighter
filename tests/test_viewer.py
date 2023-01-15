""" Test if viewer can be executed. Does not verify output """

import unittest

import torch
from matplotlib import pyplot as plt
from parameterized import parameterized

from lighter.viewer import BasicViewer, DicomExplorer, ListViewer

plt.ion()

TEST_IMAGE_3D = torch.randn(9, 9, 9)
TEST_IMAGE_2D = torch.randn(9, 9)
TEST_IMAGE_2D_RGB = torch.randn(3, 9, 9)

TEST_LABEL_3D = torch.rand(9, 9, 9).round()
TEST_LABEL_2D = torch.rand(9, 9).round()


TEST_CASE1 = [TEST_IMAGE_3D, TEST_LABEL_3D, None, None, None]
TEST_CASE2 = [TEST_IMAGE_2D, TEST_LABEL_2D, None, None, None]
TEST_CASE3 = [TEST_IMAGE_2D_RGB, TEST_LABEL_2D, None, None, "RGB"]

TEST_CASE4 = [TEST_IMAGE_3D, "Text Label", "Text Prediction", None, None]
TEST_CASE5 = [TEST_IMAGE_2D, "Text Label", "Text Prediction", None, None]
TEST_CASE6 = [TEST_IMAGE_2D_RGB, "Text Label", "Text Prediction", None, "RGB"]

TEST_CASE7 = [TEST_IMAGE_3D, "Text Label", 0.123, "Some Description", None]
TEST_CASE8 = [TEST_IMAGE_2D, 2, "Text Prediction", "Some Description", None]
TEST_CASE9 = [TEST_IMAGE_2D_RGB, 1, 1, "Some Description", "RGB"]

TEST_CASE10 = [TEST_IMAGE_3D, None, None, None, None]


TEST_CASES = [
    TEST_CASE1,
    TEST_CASE2,
    TEST_CASE3,
    TEST_CASE4,
    TEST_CASE5,
    TEST_CASE6,
    TEST_CASE7,
    TEST_CASE8,
    TEST_CASE9,
]


class TestViewer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_viewers(self, image, label, prediction, description, mode):
        """Test all viewer in order. Otherwise parallel execution might lead to blocking."""
        # pyplot windows will open and close during this test in fast succession.
        # This might lead to the windows blocking each other and the test to fail.
        # With plt.close("all") and plt.pause(0.001) as well as plt.ion() at the beginning of the file
        # it seems to work. If tests should fail on another machine, plt.show(block=False) might need to be
        # added to the viewer classes.

        basic_viewer = BasicViewer(x=image, y=label, prediction=prediction, description=description, mode=mode)
        basic_viewer.show()
        plt.close("all")
        plt.pause(0.001)

        dicom_explorer = DicomExplorer(x=image, y=label, prediction=prediction, description=description, mode=mode)
        dicom_explorer.show()
        plt.close("all")
        plt.pause(0.001)

        if label is not None:
            label = [label, label]
        if prediction is not None:
            prediction = [prediction, prediction]
        if description is not None:
            description = [description, description]

        list_viewer = ListViewer(x=[image, image], y=label, prediction=prediction, description=description, mode=mode)
        list_viewer.show()
        plt.close("all")
        plt.pause(0.001)


if __name__ == "__main__":
    unittest.main()
