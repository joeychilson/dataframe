// Package reduce provides aggregation kernels shared by series and dataframe.
package reduce

import (
	"cmp"
	"iter"
	"math"
)

// Number permits built-in real and complex numeric types, including defined
// types with those underlying types.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 | ~complex64 | ~complex128
}

// Real permits built-in integer and floating-point types, including defined
// types with those underlying types.
type Real interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// Sum returns the sum of values and whether the sequence yields any values.
func Sum[T Number](values iter.Seq2[int, T]) (T, bool) {
	var sum T
	found := false
	for _, value := range values {
		sum += value
		found = true
	}
	return sum, found
}

// Mean returns the arithmetic mean of values and whether the sequence yields
// any values.
func Mean[T Real](values iter.Seq2[int, T]) (float64, bool) {
	mean := 0.0
	count := 0
	for _, value := range values {
		mean, count = UpdateMean(mean, count, value)
	}
	if count == 0 {
		return 0, false
	}
	return mean, true
}

// UpdateMean incorporates value into the arithmetic mean of count preceding
// values and returns the updated mean and count. It avoids intermediate
// overflow and preserves same-signed infinities.
func UpdateMean[T Real](mean float64, count int, value T) (float64, int) {
	count++
	converted := float64(value)
	scale := float64(count)
	if math.IsInf(mean, 0) || math.IsInf(converted, 0) || (mean < 0) != (converted < 0) {
		return mean*(float64(count-1)/scale) + converted/scale, count
	}
	return mean + (converted-mean)/scale, count
}

// Min returns the smallest value and whether the sequence yields any values.
func Min[T cmp.Ordered](values iter.Seq2[int, T]) (T, bool) {
	var minimum T
	found := false
	for _, value := range values {
		if !found {
			minimum = value
			found = true
		} else {
			minimum = min(minimum, value)
		}
	}
	return minimum, found
}

// Max returns the largest value and whether the sequence yields any values.
func Max[T cmp.Ordered](values iter.Seq2[int, T]) (T, bool) {
	var maximum T
	found := false
	for _, value := range values {
		if !found {
			maximum = value
			found = true
		} else {
			maximum = max(maximum, value)
		}
	}
	return maximum, found
}
