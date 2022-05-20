import cv2
import os
import numpy as np

class Dataset:
    """Utility class for loading gifs datasets
    from a directory as a numpy array and flattening them for easy use
    in fitting models
    """
    def __init__(self, dir: str) -> None:
        self.dir = dir

    def load(self) -> np.ndarray:
        """Load the dataset as a numpy array

        Returns:
            np.ndarray: dataset
        """
        dataset = []
        files = os.listdir(self.dir)
        for f in files:
            dataset.append(self.__load_image(self.dir, f))
        return np.array(dataset)

    def __load_image(self, dir, f) -> np.ndarray:
        img = cv2.VideoCapture(os.path.join(dir,f)).read(0)[1]
        self.height, self.width, _ = img.shape
        # pick grayscale only
        return img[:,:,:1].flatten()

    def show_image(self, winname, image) -> None:
        """Show image in a window

        Args:
            winname (str): window name for showing
            image (np.ndarray): pixels array of the image
        """
        reshaped = image.reshape((self.height, self.width))
        cv2.imshow(winname,reshaped)
        cv2.waitKey(0)

if __name__ == "__main__":
    dataset = Dataset("./images/")
    imgs = dataset.load()
    dataset.show_image("test",imgs[0])