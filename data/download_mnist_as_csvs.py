import os


def decode_mnist_idx_files(image_filepath, label_filepath):
    # convert image and label files http://yann.lecun.com/exdb/mnist/
    # to list of images arrays and a list of labels
    #
    # images is a list of lists, each inner list is of size 28*28=784, containing
    # an integer between 0 (representing a black pixel) and 255 (representing a white
    # pixel)
    #
    # labels is a list of integers

    with open(image_filepath, "rb") as img_f, open(label_filepath, "rb") as lbl_f:
        img_f.read(16)  # first 16 bytes can be thrown away
        lbl_f.read(8)  # first 8 bytes can be thrown away

        # each byte in the label file corresponds to an example (minus the first 8
        # bytes), so we can get the number of examples in the data by reading the
        # the number of bytes in the labels file
        N = os.stat(label_filepath).st_size - 8

        images = [[ord(img_f.read(1)) for _ in range(28 * 28)] for _ in range(N)]
        labels = [ord(lbl_f.read(1)) for _ in range(N)]

        return images, labels


def convert_mnist_idx_to_csv(image_filepath, label_filepath, output_filepath):
    images, labels = decode_mnist_idx_files(image_filepath, label_filepath)

    with open(output_filepath, "w") as fo:
        for image, label in zip(images, labels):
            fo.write(",".join(map(str, [label] + image)) + "\n")


def download_mnist_as_csvs(train_csv_output_path, test_csv_output_path):
    # download compressed files
    os.system(
        "wget -O /tmp/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    )
    os.system(
        "wget -O /tmp/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    )
    os.system(
        "wget -O /tmp/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    )
    os.system(
        "wget -O /tmp/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    )

    # uncompress files (automatically deletes compressed files afterwards)
    os.system("gzip -d /tmp/train-images-idx3-ubyte.gz")
    os.system("gzip -d /tmp/train-labels-idx1-ubyte.gz")
    os.system("gzip -d /tmp/t10k-images-idx3-ubyte.gz")
    os.system("gzip -d /tmp/t10k-labels-idx1-ubyte.gz")

    # convert to csv
    convert_mnist_idx_to_csv(
        "/tmp/train-images-idx3-ubyte",
        "/tmp/train-labels-idx1-ubyte",
        train_csv_output_path,
    )
    convert_mnist_idx_to_csv(
        "/tmp/t10k-images-idx3-ubyte",
        "/tmp/t10k-labels-idx1-ubyte",
        test_csv_output_path,
    )

    # remove uncompresssed files
    os.system("rm /tmp/train-images-idx3-ubyte")
    os.system("rm /tmp/train-labels-idx1-ubyte")
    os.system("rm /tmp/t10k-images-idx3-ubyte")
    os.system("rm /tmp/t10k-labels-idx1-ubyte")


if __name__ == "__main__":
    download_mnist_as_csvs(
        os.path.join(os.path.dirname(__file__), "train.csv"),
        os.path.join(os.path.dirname(__file__), "test.csv"),
    )
