// Package series provides immutable, nullable, typed columns and the
// operations that act on them.
//
// Series is the typed column value used both independently and inside a
// dataframe. Operations align series positionally and require equal lengths
// when they combine rows. Series immutability is shallow: outer storage is not
// exposed for mutation, but references contained in element values remain
// shared.
//
// Type-changing operations such as [Series.Map], [Series.Map2], and
// [Series.Reduce] are generic methods and require Go 1.27 or newer.
package series

import (
	"errors"
	"fmt"
	"iter"
	"math"
	"slices"

	"github.com/joeychilson/dataframe/internal/bitmap"
	"github.com/joeychilson/dataframe/mask"
)

// ErrLengthMismatch is returned when a constructor receives positional inputs
// with different lengths. Operations whose signatures cannot return errors
// panic on length mismatch instead.
var ErrLengthMismatch = errors.New("series length mismatch")

// Series is an immutable, homogeneous sequence of values of type T. Each row
// is either present or null. Its zero value is an empty Series.
//
// Immutability is shallow. Constructors and accessors copy their outer slices,
// and Series operations never mutate element values. Elements containing
// slices, maps, pointers, or other references remain shared with callers, who
// are responsible for synchronizing any mutation of those values.
type Series[T any] struct {
	values   []T
	validity bitmap.Bitmap
}

// New returns a non-null Series containing a copy of values.
func New[T any](values []T) Series[T] {
	return Series[T]{values: slices.Clone(values)}
}

// NewFunc returns a non-null Series containing n values produced in ascending
// row order. It calls value once per row and panics when n is negative or value
// is nil.
func NewFunc[T any](n int, value func(int) T) Series[T] {
	if n < 0 {
		panic("series: NewFunc: negative length")
	}
	if value == nil {
		panic("series: NewFunc: nil value function")
	}
	values := make([]T, n)
	for i := range values {
		values[i] = value(i)
	}
	return Series[T]{values: values}
}

// NewNullable returns a nullable Series for which valid[i] reports whether
// values[i] is present. The result remains nullable when valid is empty or
// contains only true values. Both outer slices are copied. NewNullable returns
// an error wrapping ErrLengthMismatch when their lengths differ.
func NewNullable[T any](values []T, valid []bool) (Series[T], error) {
	if len(values) != len(valid) {
		return Series[T]{}, fmt.Errorf("%w: got %d validity entries for %d values", ErrLengthMismatch, len(valid), len(values))
	}

	return Series[T]{
		values:   slices.Clone(values),
		validity: bitmap.FromBools(valid),
	}, nil
}

// NewNullableFunc returns a nullable Series containing n cells produced in
// ascending row order. The callback's boolean result reports whether each cell
// is present. It is called once per row. The result remains nullable when n is
// zero or every cell is present. NewNullableFunc panics when n is negative or
// value is nil.
func NewNullableFunc[T any](n int, value func(int) (T, bool)) Series[T] {
	if n < 0 {
		panic("series: NewNullableFunc: negative length")
	}
	if value == nil {
		panic("series: NewNullableFunc: nil value function")
	}
	result := Series[T]{values: make([]T, n), validity: bitmap.New(n)}
	for i := range result.values {
		cell, present := value(i)
		if present {
			result.values[i] = cell
			result.validity.Set(i, true)
		}
	}
	return result
}

// FromOptionals returns a nullable Series containing the supplied optional
// cells. The result remains nullable when values is empty or contains no nulls.
func FromOptionals[T any](values []Optional[T]) Series[T] {
	physical := make([]T, len(values))
	validity := bitmap.New(len(values))
	for i, value := range values {
		if value.Valid {
			physical[i] = value.Value
			validity.Set(i, true)
		}
	}
	return Series[T]{values: physical, validity: validity}
}

// Repeat returns a non-null Series containing n copies of value. It panics
// when n is negative, matching other size-taking constructors in this API.
func Repeat[T any](value T, n int) Series[T] {
	return Series[T]{values: slices.Repeat([]T{value}, n)}
}

// Len returns the number of rows, including null rows.
func (s Series[T]) Len() int {
	return len(s.values)
}

// Nullable reports whether the Series schema permits null rows. It reports
// true for a Series constructed by NewNullable or FromOptionals even when the
// Series is empty or every row is present. Use NullCount to determine whether
// the Series currently contains any null rows.
func (s Series[T]) Nullable() bool {
	return s.validity.Initialized()
}

// NullCount returns the number of null rows.
func (s Series[T]) NullCount() int {
	if !s.validity.Initialized() {
		return 0
	}
	return s.Len() - s.validity.Count()
}

// IsValid reports whether row i is present. It panics when i is out of range.
func (s Series[T]) IsValid(i int) bool {
	_ = s.values[i]
	return !s.validity.Initialized() || s.validity.At(i)
}

// At returns row i and whether it is present. A null row returns the zero value
// of T and false. At panics when i is out of range, following ordinary slice
// indexing.
func (s Series[T]) At(i int) (T, bool) {
	value := s.values[i]
	if s.validity.Initialized() && !s.validity.At(i) {
		var zero T
		return zero, false
	}
	return value, true
}

// ValueOr returns row i when present and fallback when it is null. It panics
// when i is out of range; bounds errors are not treated as nulls.
func (s Series[T]) ValueOr(i int, fallback T) T {
	value := s.values[i]
	if s.validity.Initialized() && !s.validity.At(i) {
		return fallback
	}
	return value
}

// FirstPresent returns the first non-null value and whether one exists.
func (s Series[T]) FirstPresent() (T, bool) {
	var zero T
	if len(s.values) == 0 {
		return zero, false
	}
	if !s.validity.Initialized() {
		return s.values[0], true
	}
	for i := range s.values {
		if s.validity.At(i) {
			return s.values[i], true
		}
	}
	return zero, false
}

// LastPresent returns the last non-null value and whether one exists.
func (s Series[T]) LastPresent() (T, bool) {
	var zero T
	if len(s.values) == 0 {
		return zero, false
	}
	if !s.validity.Initialized() {
		return s.values[len(s.values)-1], true
	}
	for i, value := range slices.Backward(s.values) {
		if s.validity.At(i) {
			return value, true
		}
	}
	return zero, false
}

// Values returns a shallow copy of the physical values. Values at null rows are
// unspecified; pair this result with Validity when nulls matter. References
// contained in element values remain shared.
func (s Series[T]) Values() []T {
	return slices.Clone(s.values)
}

// Validity returns one boolean per row, where true means present. The returned
// slice is a copy. A non-null Series returns an all-true slice.
func (s Series[T]) Validity() []bool {
	if s.validity.Initialized() {
		return s.validity.Bools()
	}

	return slices.Repeat([]bool{true}, len(s.values))
}

// Optionals returns a materialized copy of all cells. Null cells contain the
// zero value of T.
func (s Series[T]) Optionals() []Optional[T] {
	values := make([]Optional[T], len(s.values))
	for i, value := range s.values {
		if !s.validity.Initialized() || s.validity.At(i) {
			values[i] = Some(value)
		}
	}
	return values
}

// All iterates every row as its index and optional value.
func (s Series[T]) All() iter.Seq2[int, Optional[T]] {
	return func(yield func(int, Optional[T]) bool) {
		for i, value := range s.values {
			cell := Optional[T]{}
			if !s.validity.Initialized() || s.validity.At(i) {
				cell = Some(value)
			}
			if !yield(i, cell) {
				return
			}
		}
	}
}

// Present iterates non-null rows as their original index and value.
func (s Series[T]) Present() iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		if s.validity.Initialized() {
			for i := range s.validity.Rows() {
				if !yield(i, s.values[i]) {
					return
				}
			}
			return
		}
		for i, value := range s.values {
			if !yield(i, value) {
				return
			}
		}
	}
}

// EqualFunc reports whether s and other have equal lengths, validity, and
// present values according to equal. It panics when equal is nil.
func (s Series[T]) EqualFunc(other Series[T], equal func(T, T) bool) bool {
	if equal == nil {
		panic("series: EqualFunc: nil equality function")
	}
	if s.Len() != other.Len() {
		return false
	}
	for i, value := range s.values {
		valid := !s.validity.Initialized() || s.validity.At(i)
		otherValid := !other.validity.Initialized() || other.validity.At(i)
		if valid != otherValid {
			return false
		}
		if valid && !equal(value, other.values[i]) {
			return false
		}
	}
	return true
}

// Equal reports whether a and b have equal lengths, validity, and present
// values according to ==. NaN therefore remains unequal to itself.
func Equal[T comparable](a, b Series[T]) bool {
	return a.EqualFunc(b, func(x, y T) bool { return x == y })
}

// Concat returns s followed by others in order. The result is nullable when s
// or any input is nullable. With no inputs, Concat returns s without copying.
// It panics if the resulting length overflows int.
func (s Series[T]) Concat(others ...Series[T]) Series[T] {
	if len(others) == 0 {
		return s
	}

	total := s.Len()
	nullable := s.validity.Initialized()
	for _, other := range others {
		if other.Len() > math.MaxInt-total {
			panic("series: Concat: length out of range")
		}
		total += other.Len()
		nullable = nullable || other.validity.Initialized()
	}

	result := Series[T]{values: make([]T, total)}
	offset := copy(result.values, s.values)
	for _, other := range others {
		offset += copy(result.values[offset:], other.values)
	}
	if !nullable {
		return result
	}

	result.validity = bitmap.Filled(total)
	offset = s.Len()
	if s.validity.Initialized() {
		result.validity.Copy(0, s.validity)
	}
	for _, other := range others {
		if other.validity.Initialized() {
			result.validity.Copy(offset, other.validity)
		}
		offset += other.Len()
	}
	return result
}

// Map applies fn to present values. Nulls propagate without calling fn, and
// the result preserves s's nullable schema. It panics when fn is nil.
func (s Series[T]) Map[U any](fn func(T) U) Series[U] {
	if fn == nil {
		panic("series: Map: nil function")
	}
	result := Series[U]{values: make([]U, s.Len())}
	if !s.validity.Initialized() {
		for i, value := range s.values {
			result.values[i] = fn(value)
		}
		return result
	}

	result.validity = s.validity.Clone()
	for i := range s.validity.Rows() {
		result.values[i] = fn(s.values[i])
	}
	return result
}

// MapCells applies fn to every cell, including null cells. It is the explicit
// escape hatch for transforms whose output validity depends on input nulls.
// The result is nullable even when every returned cell is present. It panics
// when fn is nil.
func (s Series[T]) MapCells[U any](fn func(Optional[T]) Optional[U]) Series[U] {
	if fn == nil {
		panic("series: MapCells: nil function")
	}
	result := Series[U]{
		values:   make([]U, s.Len()),
		validity: bitmap.New(s.Len()),
	}
	for i, value := range s.values {
		cell := Optional[T]{}
		if !s.validity.Initialized() || s.validity.At(i) {
			cell = Some(value)
		}
		mapped := fn(cell)
		if mapped.Valid {
			result.values[i] = mapped.Value
			result.validity.Set(i, true)
		}
	}
	return result
}

// MapOptional applies fn to present values. The boolean returned by fn reports
// whether the mapped value is present, following Go's comma-ok convention.
// Input nulls propagate without calling fn. The result is nullable even when
// every mapped value is present. It panics when fn is nil.
func (s Series[T]) MapOptional[U any](fn func(T) (U, bool)) Series[U] {
	if fn == nil {
		panic("series: MapOptional: nil function")
	}
	result := Series[U]{
		values:   make([]U, s.Len()),
		validity: bitmap.New(s.Len()),
	}
	for i, value := range s.Present() {
		mapped, valid := fn(value)
		if valid {
			result.values[i] = mapped
			result.validity.Set(i, true)
		}
	}
	return result
}

// TryMap is Map for callbacks that can fail. It stops at the first error and
// wraps it with the failing row index. It panics when fn is nil.
func (s Series[T]) TryMap[U any](fn func(T) (U, error)) (Series[U], error) {
	if fn == nil {
		panic("series: TryMap: nil function")
	}
	result := Series[U]{values: make([]U, s.Len())}
	if s.validity.Initialized() {
		result.validity = s.validity.Clone()
	}
	for i, value := range s.Present() {
		mapped, err := fn(value)
		if err != nil {
			return Series[U]{}, fmt.Errorf("series: try map: row %d: %w", i, err)
		}
		result.values[i] = mapped
	}
	return result, nil
}

// TryMapCells is MapCells for callbacks that can fail. It stops at the first
// error and wraps it with the failing row index. It panics when fn is nil.
func (s Series[T]) TryMapCells[U any](fn func(Optional[T]) (Optional[U], error)) (Series[U], error) {
	if fn == nil {
		panic("series: TryMapCells: nil function")
	}
	result := Series[U]{
		values:   make([]U, s.Len()),
		validity: bitmap.New(s.Len()),
	}
	for i, value := range s.values {
		cell := Optional[T]{}
		if !s.validity.Initialized() || s.validity.At(i) {
			cell = Some(value)
		}
		mapped, err := fn(cell)
		if err != nil {
			return Series[U]{}, fmt.Errorf("series: try map cells: row %d: %w", i, err)
		}
		if mapped.Valid {
			result.values[i] = mapped.Value
			result.validity.Set(i, true)
		}
	}
	return result, nil
}

// Map2 combines corresponding present rows of s and other. A result row is
// null when either input row is null. Map2 panics on length mismatch or when fn
// is nil.
func (s Series[T]) Map2[U, V any](other Series[U], fn func(T, U) V) Series[V] {
	if s.Len() != other.Len() {
		panic(fmt.Sprintf("series: Map2: length mismatch: left=%d right=%d", s.Len(), other.Len()))
	}
	if fn == nil {
		panic("series: Map2: nil function")
	}
	result := Series[V]{
		values:   make([]V, s.Len()),
		validity: combinedValidity(s.validity, other.validity),
	}
	if !result.validity.Initialized() {
		for i, value := range s.values {
			result.values[i] = fn(value, other.values[i])
		}
		return result
	}
	for i := range result.validity.Rows() {
		result.values[i] = fn(s.values[i], other.values[i])
	}
	return result
}

// Map2Cells combines every pair of corresponding cells, including null cells.
// It panics on length mismatch or when fn is nil.
func (s Series[T]) Map2Cells[U, V any](other Series[U], fn func(Optional[T], Optional[U]) Optional[V]) Series[V] {
	if s.Len() != other.Len() {
		panic(fmt.Sprintf("series: Map2Cells: length mismatch: left=%d right=%d", s.Len(), other.Len()))
	}
	if fn == nil {
		panic("series: Map2Cells: nil function")
	}
	result := Series[V]{
		values:   make([]V, s.Len()),
		validity: bitmap.New(s.Len()),
	}
	for i, value := range s.values {
		left := Optional[T]{}
		if !s.validity.Initialized() || s.validity.At(i) {
			left = Some(value)
		}
		right := Optional[U]{}
		if !other.validity.Initialized() || other.validity.At(i) {
			right = Some(other.values[i])
		}
		mapped := fn(left, right)
		if mapped.Valid {
			result.values[i] = mapped.Value
			result.validity.Set(i, true)
		}
	}
	return result
}

// TryMap2 is Map2 for callbacks that can fail. It stops at the first error and
// wraps it with the failing row index. It returns an error wrapping
// ErrLengthMismatch when the input lengths differ and panics when fn is nil.
func (s Series[T]) TryMap2[U, V any](other Series[U], fn func(T, U) (V, error)) (Series[V], error) {
	if s.Len() != other.Len() {
		return Series[V]{}, fmt.Errorf("%w: left=%d right=%d", ErrLengthMismatch, s.Len(), other.Len())
	}
	if fn == nil {
		panic("series: TryMap2: nil function")
	}
	result := Series[V]{
		values:   make([]V, s.Len()),
		validity: combinedValidity(s.validity, other.validity),
	}
	for i, value := range s.values {
		if result.validity.Initialized() && !result.validity.At(i) {
			continue
		}
		mapped, err := fn(value, other.values[i])
		if err != nil {
			return Series[V]{}, fmt.Errorf("series: try map 2: row %d: %w", i, err)
		}
		result.values[i] = mapped
	}
	return result, nil
}

// TryMap2Cells is Map2Cells for callbacks that can fail. It stops at the first
// error and wraps it with the failing row index. It returns an error wrapping
// ErrLengthMismatch when the input lengths differ and panics when fn is nil.
func (s Series[T]) TryMap2Cells[U, V any](other Series[U], fn func(Optional[T], Optional[U]) (Optional[V], error)) (Series[V], error) {
	if s.Len() != other.Len() {
		return Series[V]{}, fmt.Errorf("%w: left=%d right=%d", ErrLengthMismatch, s.Len(), other.Len())
	}
	if fn == nil {
		panic("series: TryMap2Cells: nil function")
	}
	result := Series[V]{
		values:   make([]V, s.Len()),
		validity: bitmap.New(s.Len()),
	}
	for i, value := range s.values {
		left := Optional[T]{}
		if !s.validity.Initialized() || s.validity.At(i) {
			left = Some(value)
		}
		right := Optional[U]{}
		if !other.validity.Initialized() || other.validity.At(i) {
			right = Some(other.values[i])
		}
		mapped, err := fn(left, right)
		if err != nil {
			return Series[V]{}, fmt.Errorf("series: try map 2 cells: row %d: %w", i, err)
		}
		if mapped.Valid {
			result.values[i] = mapped.Value
			result.validity.Set(i, true)
		}
	}
	return result, nil
}

// Scan folds present values from left to right and returns the running
// accumulations. Null rows remain null and leave the accumulator unchanged. It
// panics when fn is nil.
func (s Series[T]) Scan[R any](initial R, fn func(R, T) R) Series[R] {
	if fn == nil {
		panic("series: Scan: nil function")
	}
	result := Series[R]{values: make([]R, s.Len())}
	if s.validity.Initialized() {
		result.validity = s.validity.Clone()
	}
	accumulator := initial
	for i, value := range s.Present() {
		accumulator = fn(accumulator, value)
		result.values[i] = accumulator
	}
	return result
}

// Reduce folds present values from left to right. Nulls are skipped. It panics
// when fn is nil.
func (s Series[T]) Reduce[R any](initial R, fn func(R, T) R) R {
	if fn == nil {
		panic("series: Reduce: nil function")
	}
	result := initial
	for _, value := range s.Present() {
		result = fn(result, value)
	}
	return result
}

// SortedFunc returns a stable ordering of s using compare. Nulls sort last and
// are never passed to compare. The comparator follows cmp.Compare and
// slices.SortFunc by returning negative, zero, or positive. It panics when
// compare is nil.
func (s Series[T]) SortedFunc(compare func(T, T) int) Series[T] {
	if compare == nil {
		panic("series: SortedFunc: nil comparator")
	}
	if s.Len() < 2 {
		return s
	}
	if !s.validity.Initialized() {
		if slices.IsSortedFunc(s.values, compare) {
			return s
		}
		values := slices.Clone(s.values)
		slices.SortStableFunc(values, compare)
		return Series[T]{values: values}
	}

	compareRows := func(left, right int) int {
		leftValid := s.validity.At(left)
		rightValid := s.validity.At(right)
		switch {
		case !leftValid && !rightValid:
			return 0
		case !leftValid:
			return 1
		case !rightValid:
			return -1
		default:
			return compare(s.values[left], s.values[right])
		}
	}
	sorted := true
	for row := 1; row < s.Len(); row++ {
		if compareRows(row-1, row) > 0 {
			sorted = false
			break
		}
	}
	if sorted {
		return s
	}

	rows := make([]int, s.Len())
	for i := range rows {
		rows[i] = i
	}
	slices.SortStableFunc(rows, compareRows)
	return s.Take(rows)
}

// Filter returns rows selected by mask in their original order. The result
// preserves s's nullable schema. Filter panics on length mismatch.
func (s Series[T]) Filter(selection mask.Mask) Series[T] {
	if s.Len() != selection.Len() {
		panic(fmt.Sprintf("series: Filter: length mismatch: series=%d mask=%d", s.Len(), selection.Len()))
	}

	selected := selection.Count()
	if selected == s.Len() {
		return s
	}
	result := Series[T]{values: make([]T, selected)}
	if s.validity.Initialized() {
		result.validity = bitmap.New(selected)
	}

	i := 0
	for row := range selection.Rows() {
		result.values[i] = s.values[row]
		if result.validity.Initialized() && s.validity.At(row) {
			result.validity.Set(i, true)
		}
		i++
	}
	return result
}

// Take returns rows at the requested indexes in the supplied order. Repeated
// indexes produce repeated rows. The result preserves s's nullable schema.
// Take panics on an invalid index.
func (s Series[T]) Take(rows []int) Series[T] {
	if len(rows) == s.Len() {
		identity := true
		for i, row := range rows {
			if row != i {
				identity = false
				break
			}
		}
		if identity {
			return s
		}
	}

	result := Series[T]{values: make([]T, len(rows))}
	if s.validity.Initialized() {
		result.validity = bitmap.New(len(rows))
	}

	for i, row := range rows {
		result.values[i] = s.values[row]
		if result.validity.Initialized() && s.validity.At(row) {
			result.validity.Set(i, true)
		}
	}
	return result
}

// TakeNullable returns rows selected by nullable indexes. A null index creates
// a null result row; a present index inherits the selected source row's
// validity. The result is always nullable. TakeNullable panics when a present
// index is outside [0, Len()).
func (s Series[T]) TakeNullable(rows Series[int]) Series[T] {
	result := Series[T]{
		values:   make([]T, rows.Len()),
		validity: bitmap.New(rows.Len()),
	}
	for i, row := range rows.Present() {
		result.values[i] = s.values[row]
		if !s.validity.Initialized() || s.validity.At(row) {
			result.validity.Set(i, true)
		}
	}
	return result
}

// Head returns the first min(n, Len()) rows, preserves s's nullable schema, and
// shares storage with s. It panics when n is negative.
func (s Series[T]) Head(n int) Series[T] {
	if n < 0 {
		panic("series: Head: negative count")
	}
	return s.Slice(0, min(n, s.Len()))
}

// Tail returns the last min(n, Len()) rows, preserves s's nullable schema, and
// shares storage with s. It panics when n is negative.
func (s Series[T]) Tail(n int) Series[T] {
	if n < 0 {
		panic("series: Tail: negative count")
	}
	count := min(n, s.Len())
	return s.Slice(s.Len()-count, s.Len())
}

// Slice returns rows in the half-open interval [start, end), preserves s's
// nullable schema, and shares storage with s. It panics on invalid bounds,
// like slicing a Go slice.
func (s Series[T]) Slice(start, end int) Series[T] {
	if start < 0 || end < start || end > len(s.values) {
		panic("series: Slice: invalid bounds")
	}
	result := Series[T]{values: s.values[start:end:end]}
	if s.validity.Initialized() {
		result.validity = s.validity.Slice(start, end)
	}
	return result
}

// IsNull returns a Mask with the same length as s that selects its null rows.
func (s Series[T]) IsNull() mask.Mask {
	if !s.validity.Initialized() {
		return mask.None(s.Len())
	}
	return mask.Mask(s.validity.Not())
}

// IsNotNull returns a Mask with the same length as s that selects its present
// rows.
func (s Series[T]) IsNotNull() mask.Mask {
	if !s.validity.Initialized() {
		return mask.All(s.Len())
	}
	return mask.Mask(s.validity.Clone())
}

// FillNull replaces null rows with value and returns a non-null Series. When s
// contains no null rows, the result shares value storage with s.
func (s Series[T]) FillNull(value T) Series[T] {
	if !s.validity.Initialized() {
		return s
	}

	nullCount := s.NullCount()
	if nullCount == 0 {
		return Series[T]{values: s.values}
	}
	if nullCount == s.Len() {
		return Repeat(value, s.Len())
	}

	values := slices.Clone(s.values)
	for i := range s.validity.UnsetRows() {
		values[i] = value
	}
	return Series[T]{values: values}
}

// DropNull returns the present rows as a non-null Series. When s contains no
// null rows, the result shares value storage with s.
func (s Series[T]) DropNull() Series[T] {
	if !s.validity.Initialized() {
		return s
	}

	nullCount := s.NullCount()
	if nullCount == 0 {
		return Series[T]{values: s.values}
	}
	if nullCount == s.Len() {
		return Series[T]{}
	}

	values := make([]T, 0, s.Len()-nullCount)
	for _, value := range s.Present() {
		values = append(values, value)
	}
	return Series[T]{values: values}
}

func combinedValidity(left, right bitmap.Bitmap) bitmap.Bitmap {
	switch {
	case !left.Initialized() && !right.Initialized():
		return bitmap.Bitmap{}
	case !left.Initialized():
		return right.Clone()
	case !right.Initialized():
		return left.Clone()
	default:
		return left.And(right)
	}
}
