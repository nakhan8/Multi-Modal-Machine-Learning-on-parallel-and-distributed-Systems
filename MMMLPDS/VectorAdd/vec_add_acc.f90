program vec_add_acc
    implicit none
    integer, parameter :: n = 10000000
    real, allocatable :: a(:), b(:), c(:)
    integer :: i
    real :: start, finish

    allocate(a(n), b(n), c(n))
    a = 1.0
    b = [(real(i), i = 1, n)]

    call cpu_time(start)
    !$acc data copyin(a, b) copyout(c)
    !$acc parallel loop
    do i = 1, n
        c(i) = a(i) + b(i)
    end do
    !$acc end parallel loop
    !$acc end data
    call cpu_time(finish)

    print *, "c[0] =", c(1), ", c[n-1] =", c(n), ", elapsed =", finish - start, "seconds"
    deallocate(a, b, c)
end program vec_add_acc