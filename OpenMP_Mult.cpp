#include <iostream>
#include <algorithm>
#include <omp.h>
#include <iomanip>

using namespace std;

void GenerateMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (rand() % 10);
    }
}

void ClearMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = 0;
    }
}

void BlockMultiplyOpenMP(int* A, int* B, int* C, int n, int blockSize) {
    ClearMatrix(C, n);

#pragma omp parallel for
    for (int i0 = 0; i0 < n; i0 += blockSize) {
        for (int j0 = 0; j0 < n; j0 += blockSize) {
            for (int k0 = 0; k0 < n; k0 += blockSize) {

                for (int i = i0; i < min(i0 + blockSize, n); i++) {
                    for (int j = j0; j < min(j0 + blockSize, n); j++) {

                        int sum = 0;
                        for (int k = k0; k < min(k0 + blockSize, n); k++) {
                            sum += A[i * n + k] * B[k * n + j];
                        }

                        C[i * n + j] += sum;
                    }
                }

            }
        }
    }
}

int main() {
    int n, blockSize, numThreads;

    cout << "--- OpenMP Block Matrix Multiplication ---" << endl;
    cout << "Enter matrix size (N): ";
    cin >> n;
    cout << "Enter block size: ";
    cin >> blockSize;
    cout << "Enter number of threads: ";
    cin >> numThreads;

    omp_set_num_threads(numThreads);

    int* A = new int[n * n];
    int* B = new int[n * n];
    int* C_omp = new int[n * n];

    GenerateMatrix(A, n);
    GenerateMatrix(B, n);

    cout << "\nCalculating with " << numThreads << " threads..." << endl;

    double startOmp = omp_get_wtime();

    BlockMultiplyOpenMP(A, B, C_omp, n, blockSize);

    double endOmp = omp_get_wtime();
    double durationOmp = endOmp - startOmp;

    cout << "\nOpenMP algorithm time: " << durationOmp << " seconds" << endl;

    delete[] A;
    delete[] B;
    delete[] C_omp;

    return 0;
}