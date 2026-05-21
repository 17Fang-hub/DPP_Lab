#include <iostream>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

// Функція для генерації випадкової матриці
void GenerateMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (rand() % 10);
    }
}

// Функція для заповнення матриці нулями
void ClearMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = 0;
    }
}

// Функція для красивого виведення матриці у консоль
void PrintMatrix(const int* matrix, int rows, int cols, const std::string& name = "Matrix") {
    std::cout << "--- " << name << " ---" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Блочне множення матриць
void BlockMultiply(int* A, int* B, int* C, int n, int blockSize) {
    ClearMatrix(C, n);

    // Зовнішні 3 цикли перебирають блоки
    for (int i0 = 0; i0 < n; i0 += blockSize) {
        for (int j0 = 0; j0 < n; j0 += blockSize) {
            for (int k0 = 0; k0 < n; k0 += blockSize) {

                // Внутрішні 3 цикли перемножують конкретні блоки
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
    int n, blockSize;

    cout << "--- Sequential Block Matrix Multiplication ---" << endl;
    cout << "Enter matrix size (N): ";
    cin >> n;
    cout << "Enter block size: ";
    cin >> blockSize;

    int* A = new int[n * n];
    int* B = new int[n * n];
    int* C_std = new int[n * n];
    int* C_block = new int[n * n];

    GenerateMatrix(A, n);
    GenerateMatrix(B, n);

    cout << "\nCalculating..." << endl;

    auto startBlock = chrono::high_resolution_clock::now();
    BlockMultiply(A, B, C_block, n, blockSize);
    auto endBlock = chrono::high_resolution_clock::now();
    chrono::duration<double> durationBlock = endBlock - startBlock;

    cout << "Block algorithm time:    " << durationBlock.count() << " seconds" << endl;

    //PrintMatrix(A, n, n, "Matrix A");
    //PrintMatrix(B, n, n, "Matrix B");
    //PrintMatrix(C_std, n, n, "Result matrix C (regular algorithm)");
    //PrintMatrix(C_block, n, n, "Result Matrix C (block algorith)");

    delete[] A;
    delete[] B;
    delete[] C_std;
    delete[] C_block;

    return 0;
}