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
	source Frame
	key    series.Series[K]
	rows   [][]int
}

// GroupBy partitions f using ==. It panics when key.Len() differs from f.Len().
func (f Frame) GroupBy[K comparable](key series.Series[K]) Grouped[K] {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: GroupBy: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	grouped := Grouped[K]{source: f, key: key}
	indexes := make(map[K]int)
	nullIndex := -1
	rowGroups := make([]int, key.Len())
	groupCount := 0
	for row := 0; row < key.Len(); row++ {
		value, present := key.At(row)
		if !present {
			if nullIndex < 0 {
				nullIndex = groupCount
				groupCount++
			}
			rowGroups[row] = nullIndex
			continue
		}
		index, found := indexes[value]
		if !found {
			index = groupCount
			groupCount++
			indexes[value] = index
		}
		rowGroups[row] = index
	}
	grouped.rows = collectGroupRows(rowGroups, groupCount)
	return grouped
}

// GroupByUsing partitions f using hasher. It supports non-comparable keys and
// custom equivalence relations and panics on length mismatch or a nil hasher.
func (f Frame) GroupByUsing[K any](key series.Series[K], hasher maphash.Hasher[K]) Grouped[K] {
	if key.Len() != f.Len() {
		panic(fmt.Sprintf("dataframe: GroupByUsing: length mismatch: frame=%d key=%d", f.Len(), key.Len()))
	}
	grouped := Grouped[K]{source: f, key: key}
	indexes := hashmap.New[K, int](hasher)
	nullIndex := -1
	rowGroups := make([]int, key.Len())
	groupCount := 0
	for row := 0; row < key.Len(); row++ {
		value, present := key.At(row)
		if !present {
			if nullIndex < 0 {
				nullIndex = groupCount
				groupCount++
			}
			rowGroups[row] = nullIndex
			continue
		}
		index, found := indexes.Get(value)
		if !found {
			index = groupCount
			groupCount++
			indexes.Set(value, index)
		}
		rowGroups[row] = index
	}
	grouped.rows = collectGroupRows(rowGroups, groupCount)
	return grouped
}

// Len returns the number of groups.
func (g Grouped[K]) Len() int {
	return len(g.rows)
}

// Keys returns one key per group in first-seen order.
func (g Grouped[K]) Keys() series.Series[K] {
	rows := make([]int, len(g.rows))
	for i, group := range g.rows {
		rows[i] = group[0]
	}
	return g.key.Take(rows)
}

// Sizes returns the number of source rows in each group.
func (g Grouped[K]) Sizes() series.Series[int] {
	sizes := make([]int, len(g.rows))
	for i, rows := range g.rows {
		sizes[i] = len(rows)
	}
	return series.New(sizes)
}

// Groups iterates group keys and their rows in first-seen order.
func (g Grouped[K]) Groups() iter.Seq2[series.Optional[K], Frame] {
	return func(yield func(series.Optional[K], Frame) bool) {
		for _, rows := range g.rows {
			value, present := g.key.At(rows[0])
			key := series.None[K]()
			if present {
				key = series.Some(value)
			}
			if !yield(key, g.source.Take(rows)) {
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
// values produces a null result.
func (g Grouped[K]) Sum[T series.Number](values series.Series[T]) series.Series[T] {
	return aggregateGroups(g, values, "Sum", func(values series.Series[T], rows []int) (T, bool) {
		return reduce.Sum[T](values, rows)
	})
}

// Mean returns the arithmetic mean of present values in each group. A group
// with no present values produces a null result.
func (g Grouped[K]) Mean[T series.Real](values series.Series[T]) series.Series[float64] {
	return aggregateGroups(g, values, "Mean", func(values series.Series[T], rows []int) (float64, bool) {
		return reduce.Mean[T](values, rows)
	})
}

// Min returns the smallest present value in each group. A group with no
// present values produces a null result.
func (g Grouped[K]) Min[T cmp.Ordered](values series.Series[T]) series.Series[T] {
	return aggregateGroups(g, values, "Min", func(values series.Series[T], rows []int) (T, bool) {
		return reduce.Min[T](values, rows)
	})
}

// Max returns the largest present value in each group. A group with no present
// values produces a null result.
func (g Grouped[K]) Max[T cmp.Ordered](values series.Series[T]) series.Series[T] {
	return aggregateGroups(g, values, "Max", func(values series.Series[T], rows []int) (T, bool) {
		return reduce.Max[T](values, rows)
	})
}

// Count returns the number of present values in each group.
func (g Grouped[K]) Count[T any](values series.Series[T]) series.Series[int] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.Count: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	counts := make([]int, len(g.rows))
	for i, rows := range g.rows {
		for _, row := range rows {
			if values.IsValid(row) {
				counts[i]++
			}
		}
	}
	return series.New(counts)
}

// FirstPresent returns the first present value in each group.
func (g Grouped[K]) FirstPresent[T any](values series.Series[T]) series.Series[T] {
	return aggregateGroups(g, values, "FirstPresent", func(values series.Series[T], rows []int) (T, bool) {
		return reduce.FirstPresent[T](values, rows)
	})
}

// LastPresent returns the last present value in each group.
func (g Grouped[K]) LastPresent[T any](values series.Series[T]) series.Series[T] {
	return aggregateGroups(g, values, "LastPresent", func(values series.Series[T], rows []int) (T, bool) {
		return reduce.LastPresent[T](values, rows)
	})
}

// Map calls fn once per group with that group's values. The callback's boolean
// result reports whether the output value is present.
func (g Grouped[K]) Map[T, U any](values series.Series[T], fn func(series.Series[T]) (U, bool)) series.Series[U] {
	return aggregateGroups(g, values, "Map", func(values series.Series[T], rows []int) (U, bool) {
		return fn(values.Take(rows))
	})
}

// TryMap is Map for callbacks that can fail. It stops at the first error and
// wraps it with the zero-based group index.
func (g Grouped[K]) TryMap[T, U any](values series.Series[T], fn func(series.Series[T]) (U, bool, error)) (series.Series[U], error) {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.TryMap: length mismatch: frame=%d series=%d", g.source.Len(), values.Len()))
	}
	result := make([]U, len(g.rows))
	validity := make([]bool, len(g.rows))
	for i, rows := range g.rows {
		value, present, err := fn(values.Take(rows))
		if err != nil {
			return series.Series[U]{}, fmt.Errorf("group %d: %w", i, err)
		}
		if present {
			result[i] = value
			validity[i] = true
		}
	}
	grouped, err := series.NewNullable(result, validity)
	if err != nil {
		panic(err)
	}
	return grouped, nil
}

func collectGroupRows(rowGroups []int, groupCount int) [][]int {
	sizes := make([]int, groupCount)
	for _, group := range rowGroups {
		sizes[group]++
	}
	rows := make([][]int, groupCount)
	storage := make([]int, len(rowGroups))
	offset := 0
	for group, size := range sizes {
		rows[group] = storage[offset : offset : offset+size]
		offset += size
	}
	for row, group := range rowGroups {
		rows[group] = append(rows[group], row)
	}
	return rows
}

func aggregateGroups[K, T, U any](g Grouped[K], values series.Series[T], operation string, aggregate func(series.Series[T], []int) (U, bool)) series.Series[U] {
	if values.Len() != g.source.Len() {
		panic(fmt.Sprintf("dataframe: Grouped.%s: length mismatch: frame=%d series=%d", operation, g.source.Len(), values.Len()))
	}
	result := make([]series.Optional[U], len(g.rows))
	for i, rows := range g.rows {
		value, present := aggregate(values, rows)
		if present {
			result[i] = series.Some(value)
		}
	}
	return series.FromOptionals(result)
}
