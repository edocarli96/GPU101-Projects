## Introduction
<p align="justify">
This repository contains all the programs that you have to use as a starting point for the GPU-101 Projects SYMGS.
</p>

## Usage

### Compilation

You only need GCC to compile the various programs, which you should already have installed if you did already setup your maching for CUDA.
To compile all the examples simply type
```
make
```
Within the scope of this folder.
Note that the examples are all compiled using the -O3 flag, you have to use this flag also when compiling the GPU version of the code using nvcc.
All the parameters regarding input settings CANNOT BE CHANGED.
For symgs you should use a sparse matrix as input (the program will not run without it).
The one you should use for all of your comparisons has already been prepared for you and is available here:

https://www.dropbox.com/s/5n43vqrtm1meed1/kmer_V2a.mtx?dl=0

You can fork this repo and use this as a starting point for your project.
Remember to include all your source code as well as a PDF (4/5 pages) with your project report in the repository.
