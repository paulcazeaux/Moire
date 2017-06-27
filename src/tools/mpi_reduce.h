/* 
 * File:   mpi_reduce.h
 * Author: Paul Cazeaux
 *
 * Created on April 22, 2017, 12:28AM
 */

#ifndef moire__tools_mpi_reduce_h
#define moire__tools_mpi_reduce_h

#include <Teuchos_Comm.hpp>

namespace Utilities {
namespace MPI {

template <typename T>
void
sum(const T send [], T recv [], const int count, const Teuchos::Comm<int>& comm)
{
    Teuchos::reduceAll<int, T>(comm, Teuchos::REDUCE_SUM, count, send, recv);
};


template<>
void
sum<std::complex<float>>(const std::complex<float> send [], std::complex<float> recv [], 
                            const int count, const Teuchos::Comm<int>& comm)
{
    Teuchos::reduceAll<int, float>(comm, Teuchos::REDUCE_SUM, 2*count,
                        reinterpret_cast<const float *>(send), reinterpret_cast<float *>(recv));
}

template<>
void
sum<std::complex<double>>(const std::complex<double> send [], std::complex<double> recv [], 
                            const int count, const Teuchos::Comm<int>& comm)
{
    Teuchos::reduceAll<int, double>(comm, Teuchos::REDUCE_SUM, 2*count,
                        reinterpret_cast<const double *>(send), reinterpret_cast<double *>(recv));
}

}}

#endif
