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

        MPI.Init(args) ;

        int rank = MPI.COMM_WORLD.Rank();
        int rtotal = MPI.COMM_WORLD.Size();
        int size;
        int[] buffer = new int[1];

        if (rtotal != 8 && rtotal != 64) {
            System.err.printf("Wrong number of processes: %d\n", rtotal);
            System.exit(1);
        }

        int[][] ma = new int[0][0];
        int[][] mb = new int[0][0];

        // Distribute size to every process.
        if (rank == 0) {
            size = readSize(args[1]);
            buffer[0] = size;
            for (int i = 1; i < rtotal; i++) {
                MPI.COMM_WORLD.Send(buffer, 0, 1, MPI.INT, i, 1);
            }
        } else {
            MPI.COMM_WORLD.Recv(buffer, 0, 1, MPI.INT, 0, 1);
            size = buffer[0];
        }

        // Distribute quadrants to every process.
        // Numbers of row/col
        int n = rtotal == 8 ? 2 : 4;
        int partSize = size / n;
        int[][] a = new int[partSize][partSize];
        int[][] b = new int[partSize][partSize];
        if (rank == 0) {
            String line;
            int lineno;
            BufferedReader reader;

            // Process [row][col][i] gets [row][i] from A and [i][col] from B
            // Thus [row][col] from A goes to process [row][i][col]
            // Thus [row][col] from B goes to process [i][col][row]
            // I wish I had a better grasp of linear algebra here...
            // It's mathematically more sensible to do [row][i][col] for process,
            // but we need to sum up the array [row][col] for i, so...

            // Matrix A
            reader = new BufferedReader(new FileReader(args[1]));
            line = reader.readLine();
            lineno = 0;
            while ((line = reader.readLine()) != null) {
                int[] values = parseLine(line);
                int row = lineno / partSize;
                for (int col = 0; col < n; col++) {
                    for (int i = 0; i < n; i++) {
                        int dst = row * n * n + i * n + col;
                        if (dst == 0) {  // send to self
                            a[lineno] = Arrays.copyOf(values, partSize);
                        } else {
                            MPI.COMM_WORLD.Send(values, col * partSize, partSize, MPI.INT, dst, 1);
                        }
                    }
                }
                lineno++;
            }

            // Matrix B
            reader = new BufferedReader(new FileReader(args[2]));
            line = reader.readLine();
            lineno = 0;
            while ((line = reader.readLine()) != null) {
                int[] values = parseLine(line);
                int row = lineno / partSize;
                for (int col = 0; col < n; col++) {
                    for (int i = 0; i < n; i++) {
                        int dst = i * n * n + col * n + row;
                        if (dst == 0) {  // send to self
                            b[lineno] = Arrays.copyOf(values, partSize);
                        } else {
                            MPI.COMM_WORLD.Send(values, col * partSize, partSize, MPI.INT, dst, 1);
                        }
                    }
                }
                lineno++;
            }
        } else {
            for (int row = 0; row < partSize; row++) {
                buffer = new int[partSize];
                MPI.COMM_WORLD.Recv(buffer, 0, partSize, MPI.INT, 0, 1);
                a[row] = buffer;
                System.out.printf("%d: recv a %d\n", rank, row);
            }
            for (int row = 0; row < partSize; row++) {
                buffer = new int[partSize];
                MPI.COMM_WORLD.Recv(buffer, 0, partSize, MPI.INT, 0, 1);
                b[row] = buffer;
                System.out.printf("%d: recv b %d\n", rank, row);
            }
        }

        System.out.printf("%d: dist done\n", rank);

        // Multiply own matrices
        int[][] c;
        c = multiply(a, b);


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

            // Send to root to join
            if (rank == 0) {
                int[][][][] qc = new int[n][n][][];

                // Send to self
                qc[0][0] = c;

                // Receive from others
                for (int row = 0; row < n; row++) {
                    for (int col = 0; col < n; col++) {
                        int src = row * n * n + col * n;
                        if (src == 0) {
                            continue;
                        }
                        MPI.COMM_WORLD.Recv(buffer, 0, buffer.length, MPI.INT, src, 1);
                        qc[row][col] = inflate(partSize, buffer);
                    }
                }

                // Join
                int[][] mc = mjoin(qc);

                // Print
                printm(mc);
            } else {
                buffer = deflate(c);
                MPI.COMM_WORLD.Send(buffer, 0, buffer.length, MPI.INT, 0, 1);
            }
        } else {  // Child process
            int dst = (rank / n) * n;
            buffer = deflate(c);
            MPI.COMM_WORLD.Send(buffer, 0, buffer.length, MPI.INT, dst, 1);
        }

        MPI.Finalize();
    }

    /*
     * Multiply matrices.
     */
    static int[][] multiply(int[][] a, int[][] b) {
        return sumPartial(multPartial(a, b));
    }

    /*
     * Multiply matrices, return partial result.
     */
    static int[][][][][] multPartial(int[][] a, int[][] b) {
        int[][][][][] partial;

        // Base case.
        if (a.length == 1) {
            partial = new int[1][1][1][1][1];
            partial[0][0][0][0][0] = a[0][0] * b[0][0];
        } else {
            int[][][][] qa = msplit(a, 2);
            int[][][][] qb = msplit(b, 2);

            int n = 2;
            int[][][][] qc = new int[n][n][][];
            partial = new int[n][n][n][][];

            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    for (int i = 0; i < n; i++) {
                        partial[row][col][i] = multiply(qa[row][i], qb[i][col]);
                    }
                }
            }
        }
        return partial;
    }

    /*
     * Sum up matrix multiplication partial.
     */
    static int[][] sumPartial(int[][][][][] partial) {
        int n = partial.length;
        int[][][][] qc = new int[n][n][][];
        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                    qc[row][col] = msum(partial[row][col]);
            }
        }
        return mjoin(qc);
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
     * Split a matrix into n * n parts.
     */
    static int[][][][] msplit(int[][] m, int n) {
        int size = m.length;
        int partSize = size / n;
        int[][][][] result = new int[n][n][partSize][partSize];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i / partSize][j / partSize][i % partSize][j % partSize] = m[i][j];
            }
        }
        return result;
    }

    /*
     * Join split matrices.
     */
    static int[][] mjoin(int[][][][] m) {
        int n = m.length;
        int partSize = m[0][0].length;
        int size = partSize * n;
        int[][] result = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                result[i][j] = m[i / partSize][j / partSize][i % partSize][j % partSize];
            }
        }
        return result;
    }

    /*
     * Print matrix.
     */
    static void printm(int[][] m) {
        StringBuilder sb;
        int size = m.length;
        for (int i = 0; i < size; i++) {
            sb = new StringBuilder();
            for (int j = 0; j < size; j++) {
                sb.append(m[i][j]);
                sb.append(" ");
            }
            System.out.println(sb.toString());
        }
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
