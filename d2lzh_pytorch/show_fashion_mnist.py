def show_fashion_mnist(images,labels):
    d21.use_svg_display()
    _,figs = plt.subplots(1,len(images),figsize = (12,12))
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()