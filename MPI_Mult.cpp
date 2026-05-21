#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <iomanip>

using namespace std;

void GenerateMatrix(int* matrix, int elements) {
    for (int i = 0; i < elements; i++) {
        matrix[i] = (rand() % 10);
    }
}

void ClearMatrix(int* matrix, int elements) {
    for (int i = 0; i < elements; i++) {
        matrix[i] = 0;
    }
}

void LocalBlockMultiply(int* pProcA, int* B, int* pProcC, int RowNum, int n, int blockSize) {
    ClearMatrix(pProcC, RowNum * n);

    for (int i0 = 0; i0 < RowNum; i0 += blockSize) {
        for (int j0 = 0; j0 < n; j0 += blockSize) {
            for (int k0 = 0; k0 < n; k0 += blockSize) {

                for (int i = i0; i < min(i0 + blockSize, RowNum); i++) {
                    for (int j = j0; j < min(j0 + blockSize, n); j++) {
                        int sum = 0;
                        for (int k = k0; k < min(k0 + blockSize, n); k++) {
                            sum += pProcA[i * n + k] * B[k * n + j];
                        }
                        pProcC[i * n + j] += sum;
                    }
                }

            }
        }
    }
}

int main(int argc, char* argv[]) {
    int ProcNum, ProcRank;
    int n = 0, blockSize = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int* A = nullptr;
    int* C = nullptr;
    int* B = nullptr;
    int* pProcA = nullptr;
    int* pProcC = nullptr;

    if (ProcRank == 0) {
        cout << "--- MPI Block Matrix Multiplication ---" << endl;
        cout << "Enter matrix size (N): ";
        cin >> n;
        cout << "Enter block size: ";
        cin >> blockSize;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blockSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Масиви розподілу для MPI_Scatterv та MPI_Gatherv
    int* sendcounts = new int[ProcNum];
    int* displs = new int[ProcNum];

    int sum = 0; // Зміщення
    int remainder = n % ProcNum; // Кількість залишкових рядків

    for (int i = 0; i < ProcNum; i++) {
        int rows_for_proc_i = n / ProcNum + (i < remainder ? 1 : 0);

        sendcounts[i] = rows_for_proc_i * n;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Кількість рядків поточного процесу
    int local_rows = sendcounts[ProcRank] / n;

    B = new int[n * n];
	pProcA = new int[local_rows * n]; // Кількість рядків, які обробляє поточний процес
	pProcC = new int[local_rows * n]; // Результуюча підматриця для поточного процесу

    if (ProcRank == 0) {
        A = new int[n * n];
        C = new int[n * n];
        GenerateMatrix(A, n * n);
        GenerateMatrix(B, n * n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A, sendcounts, displs, MPI_INT,
        pProcA, sendcounts[ProcRank], MPI_INT,
        0, MPI_COMM_WORLD);

    LocalBlockMultiply(pProcA, B, pProcC, local_rows, n, blockSize);

    MPI_Gatherv(pProcC, sendcounts[ProcRank], MPI_INT,
        C, sendcounts, displs, MPI_INT,
        0, MPI_COMM_WORLD);

    double endTime = MPI_Wtime();

    if (ProcRank == 0) {
        cout << "\nMPI algorithm time on " << ProcNum << " processes: "
            << (endTime - startTime) << " seconds" << endl;
        
        //PrintMatrix(A, n, n, "Matrix A");
        //PrintMatrix(B, n, n, "Matrix B");
        //PrintMatrix(C, n, n, "Result matrix C");

        delete[] A;
        delete[] C;
    }

    delete[] B;
    delete[] pProcA;
    delete[] pProcC;
    delete[] sendcounts;
    delete[] displs;

    MPI_Finalize();
    return 0;
}