// Explicit instantiation definitions for the most common feature types.
//
// BasicDataset<T> is a template, so its code lives in Dataset.hpp.  The header
// declares these three as `extern template`, which tells every other .cpp file
// "do not generate this code yourself, it is already compiled into the
// meanshift library".  That keeps build times down and gives the static
// library real object code.  Any other type (short, long, uint8_t, ...) still
// works: it is instantiated on demand from the header.
#include "meanshift/Dataset.hpp"

namespace meanshift {

template class BasicDataset<int>;
template class BasicDataset<float>;
template class BasicDataset<double>;

} // namespace meanshift
