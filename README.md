# GPUGauss
# Language: CUDA
# Input: TXT
# Output: TSV
# Tested with: PluMA 1.0, CUDA 10

Gaussian Elimination on the GPU.

Original authors: Jesus Cabrera-Domingo, Taufiq Islam and Elio Rosabal

The plugin accepts as input a TXT file of keyword-value pairs:
matrix: TSV (tab-separated) values for the matrix
N: Matrix size (assumed N X N)

The program will then output a TSV of the matrix, in upper-triangular form
