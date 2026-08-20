// Package series provides immutable, homogeneous typed columns.
package series

import (
	"cmp"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"slices"
)

var (
	// ErrValidityRequired is returned when NewNullable is called without a
	// validity vector.
	ErrValidityRequired = errors.New("validity must not be nil")

	// ErrInvalidValidity is returned when a validity slice and its values have
	// different lengths.
	ErrInvalidValidity = errors.New("validity length must match values length")
)

// Number permits the built-in integer and floating-point types, including
// user-defined types with one of those underlying types.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// Series is an immutable, homogeneous column. A nil validity slice means that
// every value is valid; otherwise validity[i] reports whether values[i] is
// present.
//
// Series values are safe to share between Frames. Constructors and accessors
// copy caller-owned slices so a Series cannot be mutated through an alias.
type Series[T any] struct {
	values   []T
	validity []bool
}

// New constructs a non-nullable Series and copies values.
func New[T any](values []T) Series[T] {
	return Series[T]{values: slices.Clip(slices.Clone(values))}
}

// NewNullable constructs a nullable Series and copies values and validity.
func NewNullable[T any](values []T, validity []bool) (Series[T], error) {
	if validity == nil {
		return Series[T]{}, ErrValidityRequired
	}
	if len(validity) != len(values) {
		return Series[T]{}, fmt.Errorf("%w: got %d validity bits for %d values", ErrInvalidValidity, len(validity), len(values))
	}

	return Series[T]{
		values:   slices.Clip(slices.Clone(values)),
		validity: slices.Clip(slices.Clone(validity)),
	}, nil
}

// Len returns the number of rows in the Series.
func (s Series[T]) Len() int {
	return len(s.values)
}

// Type returns the exact Go type stored by the Series.
func (s Series[T]) Type() reflect.Type {
	return reflect.TypeFor[T]()
}

// Nullable reports whether the Series carries an explicit validity vector and
// can therefore represent null values. Use NullCount to determine whether it
// currently contains any nulls.
func (s Series[T]) Nullable() bool {
	return s.validity != nil
}

// Count returns the number of present values in the Series.
func (s Series[T]) Count() int {
	return len(s.values) - s.NullCount()
}

// NullCount returns the number of null values in the Series.
func (s Series[T]) NullCount() int {
	if s.validity == nil {
		return 0
	}

	n := 0
	for _, valid := range s.validity {
		if !valid {
			n++
		}
	}
	return n
}

// At returns the value at i and whether it is present. Like indexing a slice,
// At panics when i is outside [0, Len()). The value at a null row is unspecified.
func (s Series[T]) At(i int) (value T, valid bool) {
	value = s.values[i]
	return value, s.validity == nil || s.validity[i]
}

// Values returns a copy of the Series' physical values. Callers interested in
// nulls should use At or Validity as values at null positions are unspecified.
func (s Series[T]) Values() []T {
	return slices.Clone(s.values)
}

// Validity returns one boolean per row, including an all-true slice for a
// Series that has no null values.
func (s Series[T]) Validity() []bool {
	if s.validity == nil {
		return slices.Repeat([]bool{true}, len(s.values))
	}
	return slices.Clone(s.validity)
}

// Present returns an iterator over the present values in row order. Nulls are
// skipped.
func (s Series[T]) Present() iter.Seq[T] {
	if s.validity == nil {
		return slices.Values(s.values)
	}
	return func(yield func(T) bool) {
		for i, value := range s.values {
			if s.validity[i] && !yield(value) {
				return
			}
		}
	}
}

// All yields every row as (value, valid) in row order, including nulls. The
// value at a null row is unspecified.
func (s Series[T]) All() iter.Seq2[T, bool] {
	if s.validity == nil {
		return func(yield func(T, bool) bool) {
			for _, value := range s.values {
				if !yield(value, true) {
					return
				}
			}
		}
	}
	return func(yield func(T, bool) bool) {
		for i, value := range s.values {
			if !yield(value, s.validity[i]) {
				return
			}
		}
	}
}

// Each returns an iterator over the present values in row order, yielding each
// row index with its value. Nulls are skipped.
func (s Series[T]) Each() iter.Seq2[int, T] {
	if s.validity == nil {
		return func(yield func(int, T) bool) {
			for i, value := range s.values {
				if !yield(i, value) {
					return
				}
			}
		}
	}
	return func(yield func(int, T) bool) {
		for i, value := range s.values {
			if s.validity[i] && !yield(i, value) {
				return
			}
		}
	}
}

// Concat returns a Series containing s followed by each input Series in
// order. The result is nullable when any input is nullable. Concatenating no
// inputs returns s unchanged.
func (s Series[T]) Concat(parts ...Series[T]) Series[T] {
	if len(parts) == 0 {
		return s
	}

	valueParts := make([][]T, len(parts)+1)
	valueParts[0] = s.values
	nullable := s.validity != nil
	for i, part := range parts {
		valueParts[i+1] = part.values
		nullable = nullable || part.validity != nil
	}

	result := Series[T]{values: slices.Concat(valueParts...)}
	if !nullable {
		return result
	}

	result.validity = slices.Repeat([]bool{true}, len(result.values))
	if s.validity != nil {
		copy(result.validity, s.validity)
	}
	offset := len(s.values)
	for _, part := range parts {
		if part.validity != nil {
			copy(result.validity[offset:], part.validity)
		}
		offset += len(part.values)
	}

	return result
}

// Map applies fn to every present value. Nulls are propagated and fn is not
// called for them.
func (s Series[T]) Map[U any](fn func(T) U) Series[U] {
	values := make([]U, len(s.values))
	if s.validity == nil {
		for i, value := range s.values {
			values[i] = fn(value)
		}
		return Series[U]{values: values}
	}

	for i, value := range s.values {
		if s.validity[i] {
			values[i] = fn(value)
		}
	}
	return Series[U]{values: values, validity: slices.Clone(s.validity)}
}

// TryMap is Map for transforms that can fail. It stops at the first error and
// annotates it with the row index. Nulls are propagated without calling fn.
func (s Series[T]) TryMap[U any](fn func(T) (U, error)) (Series[U], error) {
	values := make([]U, len(s.values))
	if s.validity == nil {
		for i, value := range s.values {
			mapped, err := fn(value)
			if err != nil {
				return Series[U]{}, fmt.Errorf("map row %d: %w", i, err)
			}
			values[i] = mapped
		}
		return Series[U]{values: values}, nil
	}

	for i, value := range s.values {
		if !s.validity[i] {
			continue
		}

		mapped, err := fn(value)
		if err != nil {
			return Series[U]{}, fmt.Errorf("map row %d: %w", i, err)
		}
		values[i] = mapped
	}

	return Series[U]{values: values, validity: slices.Clone(s.validity)}, nil
}

// Map2 combines corresponding present values from s and other. A result row is
// null, without calling fn, when either input row is null. Map2 panics when the
// Series lengths differ.
func (s Series[T]) Map2[U, V any](other Series[U], fn func(T, U) V) Series[V] {
	if s.Len() != other.Len() {
		panic(fmt.Sprintf("series: Map2 length mismatch: left has %d rows, right has %d", s.Len(), other.Len()))
	}

	values := make([]V, s.Len())
	validity := combinedValidity(s.validity, other.validity)
	if validity == nil {
		for i, value := range s.values {
			values[i] = fn(value, other.values[i])
		}
		return Series[V]{values: values}
	}

	for i, value := range s.values {
		if validity[i] {
			values[i] = fn(value, other.values[i])
		}
	}
	return Series[V]{values: values, validity: validity}
}

// TryMap2 is Map2 for transforms that can fail. It stops at the first error
// and annotates it with the row index. Nulls are propagated without calling
// fn. TryMap2 panics when the Series lengths differ.
func (s Series[T]) TryMap2[U, V any](other Series[U], fn func(T, U) (V, error)) (Series[V], error) {
	if s.Len() != other.Len() {
		panic(fmt.Sprintf("series: TryMap2 length mismatch: left has %d rows, right has %d", s.Len(), other.Len()))
	}

	values := make([]V, s.Len())
	validity := combinedValidity(s.validity, other.validity)
	if validity == nil {
		for i, value := range s.values {
			mapped, err := fn(value, other.values[i])
			if err != nil {
				return Series[V]{}, fmt.Errorf("map row %d: %w", i, err)
			}
			values[i] = mapped
		}
		return Series[V]{values: values}, nil
	}

	for i, value := range s.values {
		if !validity[i] {
			continue
		}
		mapped, err := fn(value, other.values[i])
		if err != nil {
			return Series[V]{}, fmt.Errorf("map row %d: %w", i, err)
		}
		values[i] = mapped
	}
	return Series[V]{values: values, validity: validity}, nil
}

func combinedValidity(left, right []bool) []bool {
	switch {
	case left == nil && right == nil:
		return nil
	case left == nil:
		return slices.Clone(right)
	case right == nil:
		return slices.Clone(left)
	default:
		validity := make([]bool, len(left))
		for i := range validity {
			validity[i] = left[i] && right[i]
		}
		return validity
	}
}

// Reduce folds the present values from left to right. Nulls are skipped.
func (s Series[T]) Reduce[A any](initial A, fn func(A, T) A) A {
	acc := initial
	if s.validity == nil {
		for _, value := range s.values {
			acc = fn(acc, value)
		}
		return acc
	}
	for i, value := range s.values {
		if s.validity[i] {
			acc = fn(acc, value)
		}
	}
	return acc
}

// Sum returns the sum of the present values and whether at least one value was
// present. Arithmetic uses T and follows Go's normal overflow behavior.
func Sum[T Number](s Series[T]) (T, bool) {
	var total T
	if s.validity == nil {
		if len(s.values) == 0 {
			return total, false
		}
		for _, value := range s.values {
			total += value
		}
		return total, true
	}

	found := false
	for i, value := range s.values {
		if s.validity[i] {
			total += value
			found = true
		}
	}
	return total, found
}

// Mean returns the arithmetic mean of the present values and whether at least
// one value was present. Values are converted to float64 before summation.
func Mean[T Number](s Series[T]) (float64, bool) {
	if s.validity == nil {
		if len(s.values) == 0 {
			return 0, false
		}
		var total float64
		for _, value := range s.values {
			total += float64(value)
		}
		return total / float64(len(s.values)), true
	}

	var total float64
	count := 0
	for i, value := range s.values {
		if s.validity[i] {
			total += float64(value)
			count++
		}
	}
	if count == 0 {
		return 0, false
	}
	return total / float64(count), true
}

// Min returns the smallest present value and whether at least one value was
// present. If any present floating-point value is NaN, Min returns NaN.
func Min[T cmp.Ordered](s Series[T]) (T, bool) {
	if s.validity == nil {
		if len(s.values) == 0 {
			var zero T
			return zero, false
		}
		return slices.Min(s.values), true
	}

	var minimum T
	found := false
	for i, value := range s.values {
		if !s.validity[i] {
			continue
		}
		if !found {
			minimum = value
			found = true
			continue
		}
		minimum = min(minimum, value)
	}
	return minimum, found
}

// Max returns the largest present value and whether at least one value was
// present. If any present floating-point value is NaN, Max returns NaN.
func Max[T cmp.Ordered](s Series[T]) (T, bool) {
	if s.validity == nil {
		if len(s.values) == 0 {
			var zero T
			return zero, false
		}
		return slices.Max(s.values), true
	}

	var maximum T
	found := false
	for i, value := range s.values {
		if !s.validity[i] {
			continue
		}
		if !found {
			maximum = value
			found = true
			continue
		}
		maximum = max(maximum, value)
	}
	return maximum, found
}

// FillNull returns a non-nullable Series in which nulls have been replaced by
// value. A non-nullable Series is returned unchanged.
func (s Series[T]) FillNull(value T) Series[T] {
	if s.validity == nil {
		return s
	}

	var values []T
	for i, valid := range s.validity {
		if !valid {
			if values == nil {
				values = slices.Clone(s.values)
			}
			values[i] = value
		}
	}
	if values == nil {
		values = s.values
	}
	return Series[T]{values: values}
}

// DropNull returns a non-nullable Series containing only present values.
func (s Series[T]) DropNull() Series[T] {
	if s.validity == nil {
		return s
	}

	n := 0
	for _, valid := range s.validity {
		if valid {
			n++
		}
	}
	if n == len(s.values) {
		return Series[T]{values: s.values}
	}

	values := make([]T, 0, n)
	for i, value := range s.values {
		if s.validity[i] {
			values = append(values, value)
		}
	}
	return Series[T]{values: values}
}

// PresentRows returns the row indexes of present values in order.
func (s Series[T]) PresentRows() []int {
	if s.validity == nil {
		rows := make([]int, len(s.values))
		for row := range rows {
			rows[row] = row
		}
		return rows
	}
	rows := make([]int, 0, len(s.values))
	for row, valid := range s.validity {
		if valid {
			rows = append(rows, row)
		}
	}
	return rows
}

// MatchingRows returns the indexes of present values for which predicate
// returns true. Nulls are excluded without calling predicate.
func (s Series[T]) MatchingRows(predicate func(T) bool) []int {
	rows := make([]int, 0, len(s.values))
	if s.validity == nil {
		for i, value := range s.values {
			if predicate(value) {
				rows = append(rows, i)
			}
		}
		return rows
	}
	for i, value := range s.values {
		if s.validity[i] && predicate(value) {
			rows = append(rows, i)
		}
	}
	return rows
}

type rowKey[T comparable] struct {
	value T
	valid bool
}

// GroupRows partitions row indexes by Go equality of present values. Groups
// and their rows retain first-seen order. Nulls form one group distinct from
// the zero value of T. Because NaN is not equal to itself, each present NaN
// forms a separate group. No returned group is empty.
func GroupRows[T comparable](s Series[T]) [][]int {
	if s.validity == nil {
		positions := make(map[T]int, s.Len())
		groups := make([][]int, 0, s.Len())
		for row, value := range s.values {
			position, exists := positions[value]
			if !exists {
				position = len(groups)
				positions[value] = position
				groups = append(groups, nil)
			}
			groups[position] = append(groups[position], row)
		}
		return groups
	}

	positions := make(map[rowKey[T]]int, s.Len())
	groups := make([][]int, 0, s.Len())
	for row, value := range s.values {
		valid := s.validity[row]
		if !valid {
			var zero T
			value = zero
		}

		key := rowKey[T]{value: value, valid: valid}
		position, exists := positions[key]
		if !exists {
			position = len(groups)
			positions[key] = position
			groups = append(groups, nil)
		}
		groups[position] = append(groups[position], row)
	}

	return groups
}

// UniqueRows returns the first row index for each distinct value in s. Row
// indexes retain first-seen order, nulls form one value distinct from the zero
// value of T, and present values use Go equality, so each NaN is distinct.
func UniqueRows[T comparable](s Series[T]) []int {
	rows := make([]int, 0, s.Len())
	if s.validity == nil {
		seen := make(map[T]struct{}, s.Len())
		for row, value := range s.values {
			if _, exists := seen[value]; exists {
				continue
			}
			seen[value] = struct{}{}
			rows = append(rows, row)
		}
		return rows
	}

	seen := make(map[rowKey[T]]struct{}, s.Len())
	for row, value := range s.values {
		valid := s.validity[row]
		if !valid {
			var zero T
			value = zero
		}
		key := rowKey[T]{value: value, valid: valid}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		rows = append(rows, row)
	}
	return rows
}

// Unique returns the first occurrence of each distinct value in s. Values
// retain first-seen order, nulls form one value distinct from the zero value of
// T, and the result preserves s's nullability. Present values use Go equality,
// so each NaN is distinct.
func Unique[T comparable](s Series[T]) Series[T] {
	return s.Take(UniqueRows(s))
}

// JoinRows returns matching row-index pairs for an inner join using Go
// equality. Pairs retain left-row order and then right-row order. Nulls never
// match. Because NaN is not equal to itself, present NaNs do not match.
func JoinRows[T comparable](left, right Series[T]) (leftRows, rightRows []int) {
	rightByValue := presentRowsByValue(right)
	leftRows = make([]int, 0, left.Len())
	rightRows = make([]int, 0, left.Len())

	if left.validity == nil {
		for row, value := range left.values {
			for _, rightRow := range rightByValue[value] {
				leftRows = append(leftRows, row)
				rightRows = append(rightRows, rightRow)
			}
		}
		return leftRows, rightRows
	}

	for row, value := range left.values {
		if left.validity[row] {
			for _, rightRow := range rightByValue[value] {
				leftRows = append(leftRows, row)
				rightRows = append(rightRows, rightRow)
			}
		}
	}

	return leftRows, rightRows
}

// LeftJoinRows returns left row indexes and nullable right row indexes for a
// left join using Go equality. Rows retain left-row order and then right-row
// order. Each unmatched left row appears once with a null right row. Nulls
// never match. Because NaN is not equal to itself, present NaNs do not match.
func LeftJoinRows[T comparable](left, right Series[T]) (leftRows []int, rightRows Series[int]) {
	rightByValue := presentRowsByValue(right)
	leftRows = make([]int, 0, left.Len())
	rightRows = Series[int]{
		values:   make([]int, 0, left.Len()),
		validity: make([]bool, 0, left.Len()),
	}
	if left.validity == nil {
		for row, value := range left.values {
			matches := rightByValue[value]
			if len(matches) > 0 {
				for _, rightRow := range matches {
					leftRows = append(leftRows, row)
					rightRows.values = append(rightRows.values, rightRow)
					rightRows.validity = append(rightRows.validity, true)
				}
				continue
			}

			leftRows = append(leftRows, row)
			rightRows.values = append(rightRows.values, 0)
			rightRows.validity = append(rightRows.validity, false)
		}
		return leftRows, rightRows
	}

	for row, value := range left.values {
		if left.validity[row] {
			matches := rightByValue[value]
			if len(matches) > 0 {
				for _, rightRow := range matches {
					leftRows = append(leftRows, row)
					rightRows.values = append(rightRows.values, rightRow)
					rightRows.validity = append(rightRows.validity, true)
				}
				continue
			}
		}

		leftRows = append(leftRows, row)
		rightRows.values = append(rightRows.values, 0)
		rightRows.validity = append(rightRows.validity, false)
	}
	return leftRows, rightRows
}

func presentRowsByValue[T comparable](s Series[T]) map[T][]int {
	rows := make(map[T][]int, s.Len())
	if s.validity == nil {
		for row, value := range s.values {
			rows[value] = append(rows[value], row)
		}
		return rows
	}
	for row, value := range s.values {
		if s.validity[row] {
			rows[value] = append(rows[value], row)
		}
	}
	return rows
}

// SortedRows returns row indexes ordered by compare. The sort is stable: rows
// that compare equally retain their original order. Nulls are placed first or
// last without calling compare.
func (s Series[T]) SortedRows(compare func(T, T) int, nullsFirst bool) []int {
	if s.validity == nil {
		present := make([]int, len(s.values))
		for i := range present {
			present[i] = i
		}
		slices.SortStableFunc(present, func(left, right int) int {
			return compare(s.values[left], s.values[right])
		})
		return present
	}

	present := make([]int, 0, len(s.values))
	nulls := make([]int, 0)
	for i, valid := range s.validity {
		if valid {
			present = append(present, i)
		} else {
			nulls = append(nulls, i)
		}
	}

	slices.SortStableFunc(present, func(left, right int) int {
		return compare(s.values[left], s.values[right])
	})
	if nullsFirst {
		return append(nulls, present...)
	}
	return append(present, nulls...)
}

// Slice returns a Series containing rows in the half-open range [start:end].
// It shares immutable backing storage with s and panics when the bounds would
// be invalid for a Go slice.
func (s Series[T]) Slice(start, end int) Series[T] {
	if start < 0 || end < start || end > s.Len() {
		panic(fmt.Sprintf("series: slice bounds out of range [%d:%d] with length %d", start, end, s.Len()))
	}

	values := s.values[start:end:end]
	if s.validity == nil {
		return Series[T]{values: values}
	}
	return Series[T]{
		values:   values,
		validity: s.validity[start:end:end],
	}
}

// Take returns a Series containing the requested row indexes in order. Like
// indexing a slice, it panics if any row is outside [0, Len()).
func (s Series[T]) Take(rows []int) Series[T] {
	values := make([]T, len(rows))
	if s.validity == nil {
		for i, row := range rows {
			values[i] = s.values[row]
		}
		return Series[T]{values: values}
	}

	validity := make([]bool, len(rows))
	for i, row := range rows {
		values[i] = s.values[row]
		validity[i] = s.validity[row]
	}
	return Series[T]{values: values, validity: validity}
}

// TakeNullable returns a nullable Series selected by nullable row indexes. A
// null row index produces a null result; a present row index inherits the
// source row's validity. Like indexing a slice, it panics if any present row is
// outside [0, Len()).
func (s Series[T]) TakeNullable(rows Series[int]) Series[T] {
	values := make([]T, rows.Len())
	validity := make([]bool, rows.Len())
	for i, row := range rows.values {
		if rows.isValid(i) {
			values[i] = s.values[row]
			validity[i] = s.isValid(row)
		}
	}
	return Series[T]{values: values, validity: validity}
}

func (s Series[T]) isValid(i int) bool {
	return s.validity == nil || s.validity[i]
}
