package dataframe

import (
	"cmp"
	"fmt"
	"math"
	"math/bits"
	"slices"

	"github.com/joeychilson/dataframe/series"
)

// SortKey is an opaque, typed-at-construction positional ordering. Construct
// keys with Asc, Desc, or ByFunc.
type SortKey struct {
	values     sortValues
	reverse    bool
	nullsFirst bool
}

// Asc orders values ascending with nulls last.
func Asc[T cmp.Ordered](values series.Series[T]) SortKey {
	switch values := any(values).(type) {
	case series.Series[int]:
		return SortKey{values: primitiveSortValues[int]{values: values, encode: func(value int) uint64 {
			return uint64(uint(value)) ^ uint64(1)<<(bits.UintSize-1)
		}}}
	case series.Series[int8]:
		return SortKey{values: primitiveSortValues[int8]{values: values, encode: func(value int8) uint64 {
			return uint64(uint8(value) ^ uint8(1<<7))
		}}}
	case series.Series[int16]:
		return SortKey{values: primitiveSortValues[int16]{values: values, encode: func(value int16) uint64 {
			return uint64(uint16(value) ^ uint16(1<<15))
		}}}
	case series.Series[int32]:
		return SortKey{values: primitiveSortValues[int32]{values: values, encode: func(value int32) uint64 {
			return uint64(uint32(value) ^ uint32(1<<31))
		}}}
	case series.Series[int64]:
		return SortKey{values: primitiveSortValues[int64]{values: values, encode: func(value int64) uint64 {
			return uint64(value) ^ uint64(1<<63)
		}}}
	case series.Series[uint]:
		return SortKey{values: primitiveSortValues[uint]{values: values, encode: func(value uint) uint64 {
			return uint64(value)
		}}}
	case series.Series[uint8]:
		return SortKey{values: primitiveSortValues[uint8]{values: values, encode: func(value uint8) uint64 {
			return uint64(value)
		}}}
	case series.Series[uint16]:
		return SortKey{values: primitiveSortValues[uint16]{values: values, encode: func(value uint16) uint64 {
			return uint64(value)
		}}}
	case series.Series[uint32]:
		return SortKey{values: primitiveSortValues[uint32]{values: values, encode: func(value uint32) uint64 {
			return uint64(value)
		}}}
	case series.Series[uint64]:
		return SortKey{values: primitiveSortValues[uint64]{values: values, encode: func(value uint64) uint64 {
			return value
		}}}
	case series.Series[uintptr]:
		return SortKey{values: primitiveSortValues[uintptr]{values: values, encode: func(value uintptr) uint64 {
			return uint64(value)
		}}}
	case series.Series[float32]:
		return SortKey{values: primitiveSortValues[float32]{values: values, encode: float32SortKey}}
	case series.Series[float64]:
		return SortKey{values: primitiveSortValues[float64]{values: values, encode: float64SortKey}}
	}
	return SortKey{values: typedSortValues[T]{values: values, compare: cmp.Compare[T]}}
}

// Desc orders values descending with nulls last.
func Desc[T cmp.Ordered](values series.Series[T]) SortKey {
	return Asc(values).Reverse()
}

// ByFunc orders values ascending using compare. The comparator follows
// cmp.Compare and slices.SortFunc. ByFunc panics when compare is nil.
func ByFunc[T any](values series.Series[T], compare func(T, T) int) SortKey {
	if compare == nil {
		panic("dataframe: ByFunc: nil comparator")
	}
	return SortKey{values: typedSortValues[T]{values: values, compare: compare}}
}

// Reverse flips direction without changing explicit null placement.
func (k SortKey) Reverse() SortKey {
	k.reverse = !k.reverse
	return k
}

// NullsFirst places nulls before present values.
func (k SortKey) NullsFirst() SortKey {
	k.nullsFirst = true
	return k
}

// NullsLast places nulls after present values.
func (k SortKey) NullsLast() SortKey {
	k.nullsFirst = false
	return k
}

// SortedBy returns f stably ordered by successive keys. Earlier keys take
// precedence. It panics when a key is the zero SortKey or its length differs
// from f.Len().
func (f Frame) SortedBy(keys ...SortKey) Frame {
	for i, key := range keys {
		if key.values == nil {
			panic(fmt.Sprintf("dataframe: SortedBy: key %d is zero", i))
		}
		if key.values.len() != f.Len() {
			panic(fmt.Sprintf("dataframe: SortedBy: key %d length mismatch: frame=%d key=%d", i, f.Len(), key.values.len()))
		}
	}
	if len(keys) == 0 || f.Len() < 2 {
		return f
	}

	compareRows := func(left, right int) int {
		for _, key := range keys {
			if order := key.values.compareRows(left, right, key.reverse, key.nullsFirst); order != 0 {
				return order
			}
		}
		return 0
	}
	sorted := true
	for row := 1; row < f.Len(); row++ {
		if compareRows(row-1, row) > 0 {
			sorted = false
			break
		}
	}
	if sorted {
		return f
	}

	rows := make([]int, f.Len())
	for i := range rows {
		rows[i] = i
	}
	for _, key := range slices.Backward(keys) {
		key.values.sortStable(rows, key.reverse, key.nullsFirst)
	}
	return f.Take(rows)
}

type sortValues interface {
	len() int
	compareRows(left, right int, reverse, nullsFirst bool) int
	sortStable(rows []int, reverse, nullsFirst bool)
}

type primitiveSortValues[T cmp.Ordered] struct {
	values series.Series[T]
	encode func(T) uint64
}

func (v primitiveSortValues[T]) len() int {
	return v.values.Len()
}

func (v primitiveSortValues[T]) compareRows(left, right int, reverse, nullsFirst bool) int {
	return compareOrderedRows(v.values, left, right, reverse, nullsFirst, cmp.Compare[T])
}

func (v primitiveSortValues[T]) sortStable(rows []int, reverse, nullsFirst bool) {
	// rows is a permutation of every source index, including after earlier keys.
	keys := make([]uint64, len(rows))
	firstKey := uint64(0)
	varying := uint64(0)
	found := false
	for _, row := range rows {
		value, present := v.values.At(row)
		if !present {
			continue
		}
		key := v.encode(value)
		if reverse {
			key = ^key
		}
		keys[row] = key
		if found {
			varying |= firstKey ^ key
		} else {
			firstKey = key
			found = true
		}
	}

	current := rows
	var scratch []int
	// Least-significant-byte passes keep the input order of equal keys.
	for shift := range 8 {
		if varying&(uint64(0xff)<<(shift*8)) == 0 {
			continue
		}
		if scratch == nil {
			scratch = make([]int, len(rows))
		}
		var counts [256]int
		for _, row := range current {
			key := keys[row]
			counts[byte(key>>(shift*8))]++
		}
		offset := 0
		for value, count := range counts {
			counts[value] = offset
			offset += count
		}
		for _, row := range current {
			key := keys[row]
			value := byte(key >> (shift * 8))
			scratch[counts[value]] = row
			counts[value]++
		}
		current, scratch = scratch, current
	}

	nullCount := v.values.NullCount()
	if nullCount != 0 && nullCount != len(rows) {
		if scratch == nil {
			scratch = make([]int, len(rows))
		}
		nullOffset, presentOffset := len(rows)-nullCount, 0
		if nullsFirst {
			nullOffset, presentOffset = 0, nullCount
		}
		for _, row := range current {
			if v.values.IsValid(row) {
				scratch[presentOffset] = row
				presentOffset++
			} else {
				scratch[nullOffset] = row
				nullOffset++
			}
		}
		current = scratch
	}
	copy(rows, current)
}

func compareOrderedRows[T any](values series.Series[T], left, right int, reverse, nullsFirst bool, compare func(T, T) int) int {
	leftValue, leftPresent := values.At(left)
	rightValue, rightPresent := values.At(right)
	switch {
	case !leftPresent && !rightPresent:
		return 0
	case !leftPresent:
		if nullsFirst {
			return -1
		}
		return 1
	case !rightPresent:
		if nullsFirst {
			return 1
		}
		return -1
	case reverse:
		return compare(rightValue, leftValue)
	default:
		return compare(leftValue, rightValue)
	}
}

// cmp.Compare puts all NaNs before numbers and considers both zero signs equal.
// The remaining IEEE bits become ordered by complementing negatives and
// flipping the sign bit of non-negative values.
func float32SortKey(value float32) uint64 {
	if math.IsNaN(float64(value)) {
		return 0
	}
	if value == 0 {
		return uint64(1) << 31
	}
	encoded := math.Float32bits(value)
	if encoded&(uint32(1)<<31) != 0 {
		return uint64(^encoded)
	}
	return uint64(encoded ^ (uint32(1) << 31))
}

func float64SortKey(value float64) uint64 {
	if math.IsNaN(value) {
		return 0
	}
	if value == 0 {
		return uint64(1) << 63
	}
	encoded := math.Float64bits(value)
	if encoded&(uint64(1)<<63) != 0 {
		return ^encoded
	}
	return encoded ^ (uint64(1) << 63)
}

type typedSortValues[T any] struct {
	values  series.Series[T]
	compare func(T, T) int
}

func (v typedSortValues[T]) len() int {
	return v.values.Len()
}

func (v typedSortValues[T]) compareRows(left, right int, reverse, nullsFirst bool) int {
	return compareOrderedRows(v.values, left, right, reverse, nullsFirst, v.compare)
}

func (v typedSortValues[T]) sortStable(rows []int, reverse, nullsFirst bool) {
	slices.SortStableFunc(rows, func(left, right int) int {
		return v.compareRows(left, right, reverse, nullsFirst)
	})
}
