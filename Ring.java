/*
 * Author: Crystal Valentine
 * Feb. 2015
 * Processes pass unidirectional messages to their "neighbors".
 * This example avoids deadlock.
 */

import mpi.* ;

class Ring {
    static public void main(String[] args) throws MPIException {


        MPI.Init(args) ;

        int my_rank; // Rank of process
        int source;  // Rank of sender
        int dest;    // Rank of receiver 
        int tag=50;  // Tag for messages	
        int myrank = MPI.COMM_WORLD.Rank() ;
        int      p = MPI.COMM_WORLD.Size() ;
        int msg_size = 10000000;
        int[] token = new int[msg_size];

        if (myrank != 0) {
            int from = myrank-1;
            MPI.COMM_WORLD.Recv(token, 0, msg_size, MPI.INT, from, tag);
            System.out.println("Process " + myrank + " received token from " +
                    "process: " + from);
        } else {
            // Set te token's value if you are process 0
            for (int i = 0; i < msg_size; i++) {
                token[i] = -1;
            }
        }

        MPI.COMM_WORLD.Send(token, 0, msg_size, MPI.INT, (myrank + 1) % p, tag);

        // Now process 0 can receve from the last process
        if (myrank == 0) {
            int from = p-1;
            MPI.COMM_WORLD.Recv(token, 0, msg_size, MPI.INT, from, tag);
            System.out.println("Process 0 received token from process " +
                    from);
        }

        MPI.Finalize();
    } // main
} // class
