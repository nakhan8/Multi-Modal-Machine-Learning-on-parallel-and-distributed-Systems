program vec_add_seq
    implicit none
    integer, parameter :: N = 100000000
    real(8), allocatable :: A(:), B(:), C(:)
    integer :: i
    real(8) :: start, finish

    allocate(A(N), B(N), C(N))

    ! Initialize arrays
    A = 1.0
    B = 1.0

    ! Start timer
    call cpu_time(start)

    ! Vector addition
    do i = 1, N
        C(i) = A(i) + B(i)
    end do

    ! Stop timer
    call cpu_time(finish)

    print *, 'Sequential Vector Addition:'
    print *, 'C(1) =', C(1), ', C(N) =', C(N)
    print *, 'Total runtime (s):', finish - start

    deallocate(A, B, C)
end program vec_add_seq