class CSRMatrix:
    def __init__(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape

    def transpose(self):
        # Step 1: Convert CSR to CSC
        csc_data = []
        csc_indices = []
        csc_indptr = [0] * (self.shape[1] + 1)

        for i in range(len(self.indptr) - 1):
            start = self.indptr[i]
            end = self.indptr[i + 1]

            for j in range(start, end):
                col = self.indices[j]
                csc_data.append(self.data[j])
                csc_indices.append(i)
                csc_indptr[col + 1] += 1

        # Cumulative sum to get the final indptr for CSC
        for i in range(1, len(csc_indptr)):
            csc_indptr[i] += csc_indptr[i - 1]

        # Step 2: Convert CSC to CSR (treat it as if it were CSR)
        transposed_data = [0] * len(csc_data)
        transposed_indices = [0] * len(csc_indices)
        transposed_indptr = [0] * (self.shape[0] + 1)

        for i in range(len(csc_indptr) - 1):
            start = csc_indptr[i]
            end = csc_indptr[i + 1]

            for j in range(start, end):
                row = csc_indices[j]
                transposed_data[j] = csc_data[j]
                transposed_indices[j] = i
                transposed_indptr[row + 1] += 1

        # Cumulative sum to get the final indptr for CSR
        for i in range(1, len(transposed_indptr)):
            transposed_indptr[i] += transposed_indptr[i - 1]

        # Create the transposed CSR matrix
        transposed_csr_matrix = CSRMatrix(transposed_data, transposed_indices, transposed_indptr, self.shape[::-1])

        return transposed_csr_matrix


# Example usage:
data = [1, 2, 3, 4, 5, 6]
indices = [0, 2, 2, 0, 1, 2]
indptr = [0, 2, 3, 6]  # pointer to the start of each row
shape = (3, 3)

csr_matrix = CSRMatrix(data, indices, indptr, shape)
transposed_csr_matrix = csr_matrix.transpose()

print("Original CSR matrix:")
print(csr_matrix.data, csr_matrix.indices, csr_matrix.indptr)

print("\nTransposed CSR matrix:")
print(transposed_csr_matrix.data, transposed_csr_matrix.indices, transposed_csr_matrix.indptr)
