program vec_add_omp
    use omp_lib
    implicit none
    integer, parameter :: N = 100000000
    real(8), allocatable :: A(:), B(:), C(:)
    integer :: i
    real(8) :: start, finish

    allocate(A(N), B(N), C(N))

    call cpu_time(start)

    !$omp parallel default(shared) private(i)
        ! Initialize arrays in parallel
        !$omp do
        do i = 1, N
            A(i) = 1.0
            B(i) = 1.0
        end do
        !$omp end do

        ! Perform vector addition in parallel
        !$omp do
        do i = 1, N
            C(i) = A(i) + B(i)
        end do
        !$omp end do
    !$omp end parallel

    call cpu_time(finish)

    print *, 'C(1) =', C(1), ', C(N) =', C(N)
    print *, 'Total runtime (s):', finish - start

    deallocate(A, B, C)
end program vec_add_omp
