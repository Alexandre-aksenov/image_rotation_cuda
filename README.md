# Image rotation using Cuda

This repository shows an example implementation for rotating an image (by the angle of 1 radian).

This code has been tested on Cuda version 12.0 and Linux. The author will be interested in the necessary modifications for other platforms.

# Usage

Instructions for compiling and running in debug mode using the compiler <code>nvcc</code> are listed in <code>Makefile</code>. They can be called using:
* Create the folder for the binary (unnecessary, as this is also done by <code>build</code> below):
```bash
make target
```

* Compile:
```bash
make build
```
* Run:
```bash
make run
```
* Remove the executable:
```bash
make clean
```

# Limitations

This function keeps a constant image size before and after rotation, producing artifacts near the angles and shrinking the section with useful information. This effect becomes more serious after several successive rotations.

The parameters relative to usage (such as: image location, size, angle of rotation) are fixed in the source code. No API for providing these from outside has been added yet.

# Feedback and additional questions.

All questions about the source code should be addressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
