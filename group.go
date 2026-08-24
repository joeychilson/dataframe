package dataframe

import (
	"cmp"
	"fmt"
	"hash/maphash"
	"iter"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/internal/reduce"
	"github.com/joeychilson/dataframe/series"
)

// Grouped holds a Frame partitioned by one positional typed key. Groups retain
// first-seen order. Null keys form one group. Composite structs are the
// idiomatic way to express multi-column comparable keys. Aggregations panic
// when their values length differs from the source frame's row count.
type Grouped[K any] struct {
	source    Frame
	key       series.Series[K]
	rowGroups []int
	firstRows []int
	sizes     []int
}

// GroupBy partitions f using ==. It panics when key.Len() differs from f.Len().
func (f Frame) GroupBy[K comparable](key series.Series[K]) Grouped[K] {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: GroupBy: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	grouped := Grouped[K]{source: f, key: key, rowGroups: make([]int, key.Len())}
	indexes := make(map[K]int)
	nullIndex := -1
	groupCount := 0
	for row := range key.Len() {
		value, present := key.At(row)
		if !present {
			if nullIndex < 0 {
				nullIndex = groupCount
				groupCount++
			}
			grouped.rowGroups[row] = nullIndex
			continue
		}
		index, found := indexes[value]
		if !found {
			index = groupCount
			groupCount++
			indexes[value] = index
		}
		grouped.rowGroups[row] = index
	}
	grouped.collectMetadata(groupCount)
	return grouped
}

// GroupByUsing partitions f using hasher. It supports non-comparable keys and
// custom equivalence relations and panics on length mismatch or a nil hasher.
func (f Frame) GroupByUsing[K any](key series.Series[K], hasher maphash.Hasher[K]) Grouped[K] {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: GroupByUsing: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	grouped := Grouped[K]{source: f, key: key, rowGroups: make([]int, key.Len())}
	indexes := hashmap.New[K, int](hasher, key.Len())
	nullIndex := -1
	groupCount := 0
	for row := range key.Len() {
		value, present := key.At(row)
		if !present {
			if nullIndex < 0 {
				nullIndex = groupCount
				groupCount++
			}
			grouped.rowGroups[row] = nullIndex
			continue
		}
		index, found := indexes.Get(value)
		if !found {
			index = groupCount
			groupCount++
			indexes.Set(value, index)
		}
		grouped.rowGroups[row] = index
	}
	grouped.collectMetadata(groupCount)
	return grouped
}

// Len returns the number of groups.
func (g Grouped[K]) Len() int {
	return len(g.firstRows)
}

// Keys returns one key per group in first-seen order.
func (g Grouped[K]) Keys() series.Series[K] {
	return g.key.Take(g.firstRows)
}

// Sizes returns the number of source rows in each group.
func (g Grouped[K]) Sizes() series.Series[int] {
	return series.New(g.sizes)
}

// Groups iterates group keys and their rows in first-seen order.
func (g Grouped[K]) Groups() iter.Seq2[series.Optional[K], Frame] {
	return func(yield func(series.Optional[K], Frame) bool) {
		rows, offsets := collectGroupRows(g.rowGroups, g.sizes)
		for i := range g.Len() {
			value, present := g.key.At(g.firstRows[i])
			key := series.None[K]()
			if present {
				key = series.Some(value)
			}
			if !yield(key, g.source.Take(rows[offsets[i]:offsets[i+1]])) {
				return
			}
		}
	}
}

// Result returns a Frame whose first column is Keys under keyName, followed by
// columns. Every supplied column must contain g.Len() rows.
func (g Grouped[K]) Result(keyName string, columns ...ColumnSpec) (Frame, error) {
	result := make([]ColumnSpec, 0, len(columns)+1)
	result = append(result, ColumnFromSeries(keyName, g.Keys()))
	result = append(result, columns...)
	return New(result...)
}

// Sum returns the sum of present values in each group. A group with no present
// values produces a null result. It panics on length mismatch.
func (g Grouped[K]) Sum[T series.Number](values series.Series[T]) series.Series[T] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Sum: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	sums := make([]T, g.Len())
	present := make([]bool, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		sums[group] += value
		present[group] = true
	}
	return series.NewNullableFunc(g.Len(), func(i int) (T, bool) {
		return sums[i], present[i]
	})
}

// Mean returns the arithmetic mean of present values in each group. A group
// with no present values produces a null result. It panics on length mismatch.
func (g Grouped[K]) Mean[T series.Real](values series.Series[T]) series.Series[float64] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Mean: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	means := make([]float64, g.Len())
	counts := make([]int, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		means[group], counts[group] = reduce.UpdateMean(means[group], counts[group], value)
	}
	return series.NewNullableFunc(g.Len(), func(i int) (float64, bool) {
		return means[i], counts[i] != 0
	})
}

// Min returns the smallest present value in each group. A group with no
// present values produces a null result. It panics on length mismatch.
func (g Grouped[K]) Min[T cmp.Ordered](values series.Series[T]) series.Series[T] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Min: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	minimums := make([]T, g.Len())
	present := make([]bool, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		if present[group] {
			minimums[group] = min(minimums[group], value)
		} else {
			minimums[group] = value
			present[group] = true
		}
	}
	return series.NewNullableFunc(g.Len(), func(i int) (T, bool) {
		return minimums[i], present[i]
	})
}

// Max returns the largest present value in each group. A group with no present
// values produces a null result. It panics on length mismatch.
func (g Grouped[K]) Max[T cmp.Ordered](values series.Series[T]) series.Series[T] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Max: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	maximums := make([]T, g.Len())
	present := make([]bool, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		if present[group] {
			maximums[group] = max(maximums[group], value)
		} else {
			maximums[group] = value
			present[group] = true
		}
	}
	return series.NewNullableFunc(g.Len(), func(i int) (T, bool) {
		return maximums[i], present[i]
	})
}

// Count returns the number of present values in each group. It panics on
// length mismatch.
func (g Grouped[K]) Count[T any](values series.Series[T]) series.Series[int] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Count: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	nullCount := values.NullCount()
	if nullCount == 0 {
		return g.Sizes()
	}
	if nullCount == values.Len() {
		return series.Repeat(0, g.Len())
	}
	counts := make([]int, g.Len())
	for row := range values.Present() {
		counts[g.rowGroups[row]]++
	}
	return series.NewFunc(g.Len(), func(i int) int {
		return counts[i]
	})
}

// FirstPresent returns the first present value in each group. It panics on
// length mismatch.
func (g Grouped[K]) FirstPresent[T any](values series.Series[T]) series.Series[T] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.FirstPresent: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	first := make([]T, g.Len())
	present := make([]bool, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		if !present[group] {
			first[group] = value
			present[group] = true
		}
	}
	return series.NewNullableFunc(g.Len(), func(i int) (T, bool) {
		return first[i], present[i]
	})
}

// LastPresent returns the last present value in each group. It panics on
// length mismatch.
func (g Grouped[K]) LastPresent[T any](values series.Series[T]) series.Series[T] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.LastPresent: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	last := make([]T, g.Len())
	present := make([]bool, g.Len())
	for row, value := range values.Present() {
		group := g.rowGroups[row]
		last[group] = value
		present[group] = true
	}
	return series.NewNullableFunc(g.Len(), func(i int) (T, bool) {
		return last[i], present[i]
	})
}

// Map calls fn once per group with that group's values. The callback's boolean
// result reports whether the output value is present. Map panics on length
// mismatch or when fn is nil.
func (g Grouped[K]) Map[T, U any](values series.Series[T], fn func(series.Series[T]) (U, bool)) series.Series[U] {
	if fn == nil {
		panic("dataframe: Grouped.Map: nil function")
	}
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Map: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	rows, offsets := collectGroupRows(g.rowGroups, g.sizes)
	return series.NewNullableFunc(g.Len(), func(i int) (U, bool) {
		return fn(values.Take(rows[offsets[i]:offsets[i+1]]))
	})
}

// TryMap is Map for callbacks that can fail. It stops at the first error and
// wraps it with the zero-based group index. A values length that differs from
// the source frame returns ErrRowCount. TryMap panics when fn is nil.
func (g Grouped[K]) TryMap[T, U any](values series.Series[T], fn func(series.Series[T]) (U, bool, error)) (series.Series[U], error) {
	if values.Len() != g.source.Len() {
		return series.Series[U]{}, fmt.Errorf("%w: grouped values have %d rows, want %d", ErrRowCount, values.Len(), g.source.Len())
	}
	if fn == nil {
		panic("dataframe: Grouped.TryMap: nil function")
	}
	rows, offsets := collectGroupRows(g.rowGroups, g.sizes)
	var zero U
	var firstErr error
	result := series.NewNullableFunc(g.Len(), func(group int) (U, bool) {
		if firstErr != nil {
			return zero, false
		}
		value, present, err := fn(values.Take(rows[offsets[group]:offsets[group+1]]))
		if err != nil {
			firstErr = fmt.Errorf("group %d: %w", group, err)
			return zero, false
		}
		return value, present
	})
	if firstErr != nil {
		return series.Series[U]{}, firstErr
	}
	return result, nil
}

func (g *Grouped[K]) collectMetadata(groupCount int) {
	g.firstRows = make([]int, groupCount)
	g.sizes = make([]int, groupCount)
	for row, group := range g.rowGroups {
		if g.sizes[group] == 0 {
			g.firstRows[group] = row
		}
		g.sizes[group]++
	}
}

func collectGroupRows(rowGroups, sizes []int) ([]int, []int) {
	offsets := make([]int, len(sizes)+1)
	for group, size := range sizes {
		offsets[group+1] = offsets[group] + size
	}
	rows := make([]int, len(rowGroups))
	for row, group := range rowGroups {
		rows[offsets[group]] = row
		offsets[group]++
	}
	for group := len(sizes); group > 0; group-- {
		offsets[group] = offsets[group-1]
	}
	offsets[0] = 0
	return rows, offsets
}
