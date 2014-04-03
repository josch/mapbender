Mapbender bends a map along a path.

Example execution:

	./mapbender.py Weser-Radweg-Hauptroute.csv 0.286 6 20

The openstreetmap map of the Weser area is currently hardcoded. Upon execution
it will show the path given in Weser-Radweg-Hauptroute.csv on the map in a
matplotlib plot. It will show the approximated b-spline with the given
smoothing factor (6 above) and the map section of the given width (0.286 above)
around that curve. The area will be split into sections (20 in the example)
which will individually be transformed into rectangles which are also plotted.

On Debian systems you need the following packages:

	apt-get install python python-pil python-scipy python-tk python-matplotlib python-numpy
