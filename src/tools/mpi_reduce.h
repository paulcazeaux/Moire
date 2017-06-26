/* 
 * File:   mpi_reduce.h
 * Author: Paul Cazeaux
 *
 * Created on April 22, 2017, 12:28AM
 */

#ifndef moire__tools_mpi_reduce_h
#define moire__tools_mpi_reduce_h

#include <Teuchos_Comm.hpp>

namespace MPI_reduce {

template <typename T>
void
sum(const T send [], T recv [], 
                                const int count, const int root, 
                                const Teuchos::Comm<int>& comm)
{
    Teuchos::reduce<int, T>(send, recv, count, Teuchos::REDUCE_SUM, 0, comm);
};


template<>
void
sum<std::complex<float>>(const std::complex<float> send [], std::complex<float> recv [], 
                                const int count, const int root, 
                                const Teuchos::Comm<int>& comm)
{
    Teuchos::reduce<int, float>(reinterpret_cast<const float *>(send), reinterpret_cast<float *>(recv), 
                                    2*count, Teuchos::REDUCE_SUM, 0, comm);
}

template<>
void
sum<std::complex<double>>(const std::complex<double> send [], std::complex<double> recv [], 
                                const int count, const int root, 
                                const Teuchos::Comm<int>& comm)
{
    Teuchos::reduce<int, double>(reinterpret_cast<const double *>(send), reinterpret_cast<double *>(recv), 
                                    2*count, Teuchos::REDUCE_SUM, 0, comm);
}

}

#endif
