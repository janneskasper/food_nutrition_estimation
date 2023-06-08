import numpy as np
import cv2
from src.ellipse_detection.segment_detector import SegmentDetector
from src.ellipse_detection.ellipse import Ellipse
from src.ellipse_detection.ellipse_candidate_maker import EllipseCandidateMaker
from src.ellipse_detection.ellipse_estimator import EllipseEstimator
from src.ellipse_detection.ellipse_merger import EllipseMerger


class EllipseDetector(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.seg_detector = SegmentDetector()
        self.ellipse_cand_maker = EllipseCandidateMaker()
        self.ellipse_estimator = EllipseEstimator()
        self.ellipse_merger = EllipseMerger(input_shape[1], input_shape[0])

    def detect(self, input_image):
        """Detect ellipse from image.

        Args:
            input_image: Input image path or image array.

        Returns:
            Array of Ellipse instance that was detected from image.
        """

        # Load and convert image to grayscale
        if isinstance(input_image, str):
            image = cv2.imread(input_image, cv2.IMREAD_COLOR)
        else:
            image = input_image
        image = cv2.resize(image, (int(self.input_shape[1]),
                                   int(self.input_shape[0])))
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # Author's mistake??

        segments = self.seg_detector.detect(image)
        ellipse_cands = self.ellipse_cand_maker.make(segments)
        ellipses = self.ellipse_estimator.estimate(ellipse_cands)
        ellipses = self.ellipse_merger.merge(ellipses)

        # Return the best-fitting ellipse parameters
        best_fit_ellipse = Ellipse(np.zeros(2), 0, 0, 0)
        for ellipse in ellipses:
            if ellipse.accuracy_score > best_fit_ellipse.accuracy_score:
                best_fit_ellipse = ellipse

        return (best_fit_ellipse.center[0], best_fit_ellipse.center[1],
                best_fit_ellipse.major_len, best_fit_ellipse.minor_len,
                best_fit_ellipse.angle)
