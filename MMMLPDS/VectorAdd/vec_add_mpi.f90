program vec_add_mpi
    use mpi
    implicit none

    integer, parameter :: n = 10000000       ! total vector size
    double precision, allocatable :: a(:), b(:), c(:)
    double precision, allocatable :: local_a(:), local_b(:), local_c(:)
    integer :: ierr, rank, size
    integer :: i, chunk, start_idx, end_idx
    double precision :: t_start, t_end, elapsed_local, elapsed_max

    ! Initialize MPI
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    ! Determine chunk size per process
    chunk = n / size
    start_idx = rank * chunk + 1
    end_idx   = start_idx + chunk - 1

    ! Allocate local arrays
    allocate(local_a(chunk), local_b(chunk), local_c(chunk))

    ! Initialize local arrays
    local_a = 1.0d0
    local_b = dble([(i, i=start_idx,end_idx)])

    ! Synchronize processes before timing
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    t_start = MPI_Wtime()

    ! Vector addition
    do i = 1, chunk
        local_c(i) = local_a(i) + local_b(i)
    end do

    ! Synchronize processes after computation
    call MPI_Barrier(MPI_COMM_WORLD, ierr)
    t_end = MPI_Wtime()
    elapsed_local = t_end - t_start

    ! Reduce to get maximum elapsed time across all processes
    call MPI_Reduce(elapsed_local, elapsed_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)

    ! Allocate full vector on root process to gather results
    if (rank == 0) then
        allocate(a(n), b(n), c(n))
    end if

    ! Gather results from all processes
    call MPI_Gather(local_a, chunk, MPI_DOUBLE_PRECISION, a, chunk, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    call MPI_Gather(local_b, chunk, MPI_DOUBLE_PRECISION, b, chunk, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    call MPI_Gather(local_c, chunk, MPI_DOUBLE_PRECISION, c, chunk, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

    ! Print results from root
    if (rank == 0) then
        print *, "c[0] =", c(1), ", c[n-1] =", c(n), ", elapsed =", elapsed_max, "seconds"
    end if

    ! Clean up
    deallocate(local_a, local_b, local_c)
    if (rank == 0) then
        deallocate(a, b, c)
    end if

    call MPI_Finalize(ierr)
end program vec_add_mpi
