def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal',
                  'shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]