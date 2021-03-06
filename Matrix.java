import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.StringBuilder;
import java.util.Arrays;

import mpi.* ;

class Matrix {
    static public void main(String[] args) throws MPIException,
                                                  FileNotFoundException,
                                                  IOException {

        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int rtotal = MPI.COMM_WORLD.Size();
        int size;
        int[] buffer = new int[1];

        if (rtotal != 8 && rtotal != 64) {
            System.err.printf("Wrong number of processes: %d\n", rtotal);
            System.exit(1);
        }

        size = readSize(args[1]);

        System.err.printf("%d: size\n", rank);

        // Distribute quadrants to every process.
        // Numbers of row/col
        int n = (rtotal == 8) ? 2 : 4;
        int partSize = size / n;
        int[][] a = new int[partSize][partSize];
        int[][] b = new int[partSize][partSize];

        String line;
        int lineno;
        int curRow;
        BufferedReader reader;

        // Process [row][col][i] gets [row][i] from A and [i][col] from B
        // Thus [row][col] from A goes to process [row][i][col]
        // Thus [row][col] from B goes to process [i][col][row]
        // I wish I had a better grasp of linear algebra here...
        // It's mathematically more sensible to do [row][i][col] for process,
        // but we need to sum up the array [row][col] over i, so...

        // Matrix A
        reader = new BufferedReader(new FileReader(args[1]));
        lineno = 0;
        curRow = 0;
        while ((line = reader.readLine()) != null) {
            buffer = parseLine(line);
            int row = lineno / partSize;
            for (int col = 0; col < n; col++) {
                for (int i = 0; i < n; i++) {
                    int dst = row * n * n + i * n + col;
                    if (dst == rank) {  // send to self
                        for (int x = 0; x < partSize; x++) {
                            a[curRow][x] = buffer[col * partSize + x];
                        }
                        curRow++;
                    }
                }
            }
            lineno++;
        }

        System.err.printf("%d: read a\n", rank);

        // Matrix B
        reader = new BufferedReader(new FileReader(args[2]));
        lineno = 0;
        curRow = 0;
        while ((line = reader.readLine()) != null) {
            buffer = parseLine(line);
            int row = lineno / partSize;
            for (int col = 0; col < n; col++) {
                for (int i = 0; i < n; i++) {
                    int dst = i * n * n + col * n + row;
                    if (dst == rank) {  // send to self
                        for (int x = 0; x < partSize; x++) {
                            b[curRow][x] = buffer[col * partSize + x];
                        }
                        curRow++;
                    }
                }
            }
            lineno++;
        }

        System.err.printf("%d: read b\n", rank);

        // Multiply own matrices
        int[][] c;
        c = multiply(a, b);

        System.err.printf("%d: mult\n", rank);

        // Send partial solutions to sum processes
        buffer = new int[partSize * partSize];
        if ((rank % n) == 0) {  // sum process
            int[][][] partial = new int[n][][];

            // Send to self
            partial[0] = c;

            // Receive results
            for (int i = 1; i < n; i++) {
                MPI.COMM_WORLD.Recv(buffer, 0, buffer.length, MPI.INT, rank + i, 1);

                partial[i] = inflate(partSize, buffer);
            }

            // Sum up matrices
            c = msum(partial);

            // Send to root to print
            if (rank == 0) {
                for (lineno = 0; lineno < size; lineno++) {
                    int row = lineno / partSize;
                    buffer = new int[size];

                    // Receive all parts of line
                    for (int col = 0; col < n; col++) {
                        int src = row * n * n + col * n;
                        if (src == 0) {
                            for (int i = 0; i < partSize; i++) {
                                buffer[i] = c[lineno][i];
                            }
                        } else {
                            MPI.COMM_WORLD.Recv(buffer, col * partSize, partSize, MPI.INT, src, 1);
                        }
                    }

                    // Print line
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < buffer.length; i++) {
                        if (i != 0) {
                            sb.append(" ");
                        }
                        sb.append(buffer[i]);
                    }
                    System.out.println(sb.toString());
                }
            } else {  // sum process send to root
                buffer = deflate(c);
                for (int i = 0; i < partSize; i++) {
                    MPI.COMM_WORLD.Send(buffer, i * partSize, partSize, MPI.INT, 0, 1);
                }
                System.err.printf("%d: sum done\n", rank);
            }
        } else {  // Child process
            int dst = (rank / n) * n;
            buffer = deflate(c);
            MPI.COMM_WORLD.Send(buffer, 0, buffer.length, MPI.INT, dst, 1);
            System.err.printf("%d: child done\n", rank);
        }

        MPI.Finalize();
    }

    /*
     * Multiply matrices.
     */
    static int[][] multiply(int[][] a, int[][] b) {
        int n = a.length;
        int[][] c = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                c[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }

    /*
     * Sum up an array of matrices.
     */
    static int[][] msum(int[][][] m) {
        int n = m.length;
        int size = m[0].length;
        int[][] c = new int [size][size];
        for (int row = 0; row < size; row++) {
            for (int col = 0; col < size; col++) {
                c[row][col] = 0;
                for (int i = 0; i < n; i++) {
                    c[row][col] += m[i][row][col];
                }
            }
        }
        return c;
    }

    /*
     * Restore array to matrix.
     */
    static int[][] inflate(int size, int[] m) {
        int[][] result = new int [size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = m[i * size + j];
            }
        }
        return result;
    }

    /*
     * Flatten matrix into array.
     */
    static int[] deflate(int[][] m) {
        int size = m.length;
        int[] result = new int[size * size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i * size + j] = m[i][j];
            }
        }
        return result;
    }

    /*
     * Get size of file matrix.
     */
    static int readSize(String file) throws FileNotFoundException,
                                                IOException {
        BufferedReader reader;
        reader = new BufferedReader(new FileReader(file));

        // Determine size
        String line = reader.readLine();
        return readTokens(line).length;
    }

    /*
     * Parse values in line.
     */
    static int[] parseLine(String line) {
        return parseTokens(readTokens(line));
    }

    /*
     * Read tokens from string
     */
    static String[] readTokens(String line) {
        return line.split(" ");
    }

    /*
     * Parse String array of tokens to int array.
     */
    static int[] parseTokens(String[] tokens) {
        int[] result = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            result[i] = Integer.parseInt(tokens[i]);
        }
        return result;
    }
}
