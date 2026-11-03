// Explicit instantiation definitions for the most common input types.
// See the comment in Dataset.cpp: the template code lives in MeanShift.hpp,
// and these three versions are compiled once into the library.
#include "meanshift/MeanShift.hpp"

namespace meanshift {

template class BasicMeanShift<int>;     // int input, double compute type
template class BasicMeanShift<float>;   // float input, float compute type
template class BasicMeanShift<double>;  // the classic `MeanShift`

} // namespace meanshift
