package dataframe

import (
	"cmp"
	"fmt"
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
	return ByFunc(values, cmp.Compare[T])
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

	rows := make([]int, f.Len())
	for i := range rows {
		rows[i] = i
	}
	slices.SortStableFunc(rows, func(left, right int) int {
		for _, key := range keys {
			if order := key.values.compareRows(left, right, key.reverse, key.nullsFirst); order != 0 {
				return order
			}
		}
		return 0
	})
	return f.Take(rows)
}

type sortValues interface {
	len() int
	compareRows(left, right int, reverse, nullsFirst bool) int
}

type typedSortValues[T any] struct {
	values  series.Series[T]
	compare func(T, T) int
}

func (v typedSortValues[T]) len() int {
	return v.values.Len()
}

func (v typedSortValues[T]) compareRows(left, right int, reverse, nullsFirst bool) int {
	leftValue, leftPresent := v.values.At(left)
	rightValue, rightPresent := v.values.At(right)
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
		return v.compare(rightValue, leftValue)
	default:
		return v.compare(leftValue, rightValue)
	}
}
