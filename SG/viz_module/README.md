# Visualize WSI on local server

Use [vips](https://github.com/libvips/libvips) to create a pyramid tiff file from a given DAPI WSI tiff file.
```
$ vips tiffsave ${img} ${pyramid} --tile --tile-width=256 --tile-height=256 --pyramid
```

Use the [OpenSlide Python](https://github.com/openslide/openslide-python) interface to view the WSI.
The deepzoom_multiserver.py script starts a web interface on port 5000 and displays the image files at the specified file system location.
If image files exist in subdirectories, they will also be displayed in the list of available slides.
If this viewing application is installed on a server that also hosts the whole-slide image repository, this offers a convenient mechanism for you to view the slides without requiring local storage space.
```
$ path/to/python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 path/to/WSI/directory
```