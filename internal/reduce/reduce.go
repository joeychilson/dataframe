// Package reduce provides aggregation kernels shared by series and dataframe.
package reduce

import (
	"cmp"
	"slices"
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

// Cells is the read-only sequence required by the reduction kernels.
type Cells[T any] interface {
	Len() int
	At(int) (T, bool)
}

// Sum returns the sum of present cells selected by rows and whether any are
// present. A nil rows slice selects every cell.
func Sum[T Number, S Cells[T]](values S, rows []int) (T, bool) {
	var sum T
	found := false
	if rows == nil {
		for row := 0; row < values.Len(); row++ {
			if value, present := values.At(row); present {
				sum += value
				found = true
			}
		}
		return sum, found
	}
	for _, row := range rows {
		if value, present := values.At(row); present {
			sum += value
			found = true
		}
	}
	return sum, found
}

// Mean returns the arithmetic mean of present cells selected by rows and
// whether any are present. A nil rows slice selects every cell.
func Mean[T Real, S Cells[T]](values S, rows []int) (float64, bool) {
	sum := 0.0
	count := 0
	if rows == nil {
		for row := 0; row < values.Len(); row++ {
			if value, present := values.At(row); present {
				sum += float64(value)
				count++
			}
		}
	} else {
		for _, row := range rows {
			if value, present := values.At(row); present {
				sum += float64(value)
				count++
			}
		}
	}
	if count == 0 {
		return 0, false
	}
	return sum / float64(count), true
}

// Min returns the smallest present cell selected by rows and whether any are
// present. A nil rows slice selects every cell.
func Min[T cmp.Ordered, S Cells[T]](values S, rows []int) (T, bool) {
	var minimum T
	found := false
	if rows == nil {
		for row := 0; row < values.Len(); row++ {
			value, present := values.At(row)
			if !present {
				continue
			}
			if !found {
				minimum = value
				found = true
			} else {
				minimum = min(minimum, value)
			}
		}
		return minimum, found
	}
	for _, row := range rows {
		value, present := values.At(row)
		if !present {
			continue
		}
		if !found {
			minimum = value
			found = true
		} else {
			minimum = min(minimum, value)
		}
	}
	return minimum, found
}

// Max returns the largest present cell selected by rows and whether any are
// present. A nil rows slice selects every cell.
func Max[T cmp.Ordered, S Cells[T]](values S, rows []int) (T, bool) {
	var maximum T
	found := false
	if rows == nil {
		for row := 0; row < values.Len(); row++ {
			value, present := values.At(row)
			if !present {
				continue
			}
			if !found {
				maximum = value
				found = true
			} else {
				maximum = max(maximum, value)
			}
		}
		return maximum, found
	}
	for _, row := range rows {
		value, present := values.At(row)
		if !present {
			continue
		}
		if !found {
			maximum = value
			found = true
		} else {
			maximum = max(maximum, value)
		}
	}
	return maximum, found
}

// FirstPresent returns the first present cell selected by rows and whether one
// exists. A nil rows slice selects every cell.
func FirstPresent[T any, S Cells[T]](values S, rows []int) (T, bool) {
	if rows == nil {
		for row := 0; row < values.Len(); row++ {
			if value, present := values.At(row); present {
				return value, true
			}
		}
	} else {
		for _, row := range rows {
			if value, present := values.At(row); present {
				return value, true
			}
		}
	}
	var zero T
	return zero, false
}

// LastPresent returns the last present cell selected by rows and whether one
// exists. A nil rows slice selects every cell.
func LastPresent[T any, S Cells[T]](values S, rows []int) (T, bool) {
	if rows == nil {
		for row := values.Len() - 1; row >= 0; row-- {
			if value, present := values.At(row); present {
				return value, true
			}
		}
	} else {
		for _, row := range slices.Backward(rows) {
			if value, present := values.At(row); present {
				return value, true
			}
		}
	}
	var zero T
	return zero, false
}
