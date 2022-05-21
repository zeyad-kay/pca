import cv2
import os
import numpy as np

class Dataset:
    """Utility class for loading gifs datasets
    from a directory as a numpy array and flattening them for easy use
    in fitting models
    """
    def __init__(self, dir: str, ) -> None:
        self.dir = dir
        self.images = None
        self.labels = None

    def load(self) -> tuple[np.ndarray,np.ndarray]:
        """Load images and labels as numpy arrays

        Returns:
            tuple[np.ndarray,np.ndarray]: images and labels
        """
        dataset = []
        labels = []
        files = os.listdir(self.dir)
        for f in files:
            dataset.append(self.__load_image(self.dir, f))
            labels.append(f.split(".")[1])
        self.images, self.labels = np.array(dataset), np.array(labels)
        return self.images, self.labels

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

    def encode_labels(self) -> np.ndarray:
        """Encode labels to ints

        Returns:
            np.ndarray: int labels
        """
        if self.labels is None:
            raise ValueError("Load dataset first.")

        encoded = []
        unique = {}
        for i in range(self.labels.shape[0]):
            if not unique.get(self.labels[i]):
                unique[self.labels[i]] = i
            encoded.append(unique[self.labels[i]])
        return np.array(encoded)

if __name__ == "__main__":
    dataset = Dataset("./images/")
    imgs,lbls = dataset.load()
    encoded_lbls = dataset.encode_labels()
    dataset.show_image("test",imgs[0])