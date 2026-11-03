#ifndef MEANSHIFT_CONCEPTS_HPP
#define MEANSHIFT_CONCEPTS_HPP

// ---------------------------------------------------------------------------
// Concepts and type helpers shared by every template in the library.
//
// Why a separate header?  Dataset, MeanShift, the kernels and the tests all
// need the same answer to "which types count as numbers?".  Writing that rule
// once here means there is exactly one place to change it.
// ---------------------------------------------------------------------------

#include <concepts>
#include <type_traits>

namespace meanshift {

// ----- Character types ------------------------------------------------------
// char, wchar_t, char8_t, char16_t and char32_t are "arithmetic" in C++, but
// they hold text, not numbers.  signed char / unsigned char are NOT listed here
// because std::int8_t / std::uint8_t are usually aliases for them, and those
// are legitimate small-integer feature types.
template <class T>
concept CharacterType =
    std::same_as<std::remove_cv_t<T>, char>     ||
    std::same_as<std::remove_cv_t<T>, wchar_t>  ||
    std::same_as<std::remove_cv_t<T>, char8_t>  ||
    std::same_as<std::remove_cv_t<T>, char16_t> ||
    std::same_as<std::remove_cv_t<T>, char32_t>;

// ----- Numeric --------------------------------------------------------------
// Any built-in number we are willing to store as a feature value:
//   short, int, long, long long (signed and unsigned), int8_t/uint8_t,
//   float, double, long double.
// Rejected: bool (true/false is not a measurement), character types,
// std::string, pointers, user classes.
template <class T>
concept Numeric =
    std::is_arithmetic_v<T> &&
    !std::same_as<std::remove_cv_t<T>, bool> &&
    !CharacterType<T>;

// ----- RealNumber -----------------------------------------------------------
// The type the algorithm *computes* in.  Means, distances, kernel weights and
// standardised values are fractional, so they need a floating-point type.
template <class T>
concept RealNumber = std::floating_point<T>;

// ----- default_real_t -------------------------------------------------------
// Picks the compute type for an input type T:
//   float        -> float        (keep the user's precision choice)
//   double       -> double
//   long double  -> long double
//   any integer  -> double       (an average of ints is not an int)
template <Numeric T>
using default_real_t = std::conditional_t<std::floating_point<T>, T, double>;

// ----- KernelFunction -------------------------------------------------------
// A kernel is any callable object k such that k(distance, bandwidth) returns
// something convertible to R.  This lets users plug in their own kernel
// without inheriting from a base class (static polymorphism).
template <class K, class R>
concept KernelFunction =
    RealNumber<R> &&
    std::default_initializable<K> &&
    requires(const K k, R distance, R bandwidth) {
        { k(distance, bandwidth) } -> std::convertible_to<R>;
    };

} // namespace meanshift

#endif // MEANSHIFT_CONCEPTS_HPP
