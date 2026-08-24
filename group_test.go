package dataframe

import (
	"errors"
	"hash/maphash"
	"math"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/series"
)

func TestGroupBy_PreservesFirstKeyOrderAndPartitionsRows(t *testing.T) {
	keys := series.FromOptionals([]series.Optional[string]{
		series.Some("b"),
		series.None[string](),
		series.Some("a"),
		series.Some("b"),
		series.None[string](),
	})
	frame, err := New(Column("id", []int{0, 1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(keys)
	if grouped.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", grouped.Len())
	}
	if got := grouped.Keys().Optionals(); !reflect.DeepEqual(got, []series.Optional[string]{series.Some("b"), series.None[string](), series.Some("a")}) {
		t.Fatalf("Keys() = %#v", got)
	}
	if got := grouped.Sizes().Values(); !slices.Equal(got, []int{2, 2, 1}) {
		t.Fatalf("Sizes() = %v", got)
	}

	var groupRows [][]int
	for _, group := range grouped.Groups() {
		ids, columnErr := group.Column[int]("id")
		if columnErr != nil {
			t.Fatal(columnErr)
		}
		groupRows = append(groupRows, ids.Values())
	}
	wantRows := [][]int{{0, 3}, {1, 4}, {2}}
	if !reflect.DeepEqual(groupRows, wantRows) {
		t.Fatalf("Groups rows = %v, want %v", groupRows, wantRows)
	}
}

func TestGroupByUsing_AppliesCustomEquivalence(t *testing.T) {
	frame, err := New(Column("id", []int{0, 1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	keys := series.New([][]int{{1, 2}, {1, 2}, {3}})
	grouped := frame.GroupByUsing(keys, intSliceHasher{})
	if grouped.Len() != 2 || !slices.Equal(grouped.Sizes().Values(), []int{2, 1}) {
		t.Fatalf("groups = %d, sizes = %v", grouped.Len(), grouped.Sizes().Values())
	}
}

func TestGroupByPanicsOnInvalidArguments(t *testing.T) {
	frame, err := New(Column("id", []int{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		call func()
	}{
		{name: "GroupBy rejects a mismatched key length", call: func() { frame.GroupBy(series.New([]int{1})) }},
		{name: "GroupByUsing rejects a mismatched key length", call: func() { frame.GroupByUsing(series.New([]int{1}), maphash.ComparableHasher[int]{}) }},
		{name: "GroupByUsing rejects a nil hasher", call: func() { frame.GroupByUsing(series.New([]int{1, 2}), nil) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("function did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestGroupedAggregations_ProduceOneValuePerGroup(t *testing.T) {
	frame, err := New(Column("id", []int{0, 1, 2, 3, 4}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]string{"a", "b", "a", "b", "c"}))
	values := series.FromOptionals([]series.Optional[int]{
		series.Some(2),
		series.None[int](),
		series.Some(4),
		series.Some(8),
		series.None[int](),
	})

	if got := grouped.Sum(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(6), series.Some(8), series.None[int]()}) {
		t.Fatalf("Sum = %#v", got)
	}
	if got := grouped.Mean(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[float64]{series.Some(3.0), series.Some(8.0), series.None[float64]()}) {
		t.Fatalf("Mean = %#v", got)
	}
	if got := grouped.Min(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(2), series.Some(8), series.None[int]()}) {
		t.Fatalf("Min = %#v", got)
	}
	if got := grouped.Max(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(4), series.Some(8), series.None[int]()}) {
		t.Fatalf("Max = %#v", got)
	}
	if got := grouped.Count(values).Values(); !slices.Equal(got, []int{2, 1, 0}) {
		t.Fatalf("Count = %v", got)
	}
	if got := grouped.Count(series.New([]int{1, 2, 3, 4, 5})).Values(); !slices.Equal(got, []int{2, 2, 1}) {
		t.Fatalf("Count(non-null) = %v", got)
	}
	allPresent, err := series.NewNullable([]int{1, 2, 3, 4, 5}, []bool{true, true, true, true, true})
	if err != nil {
		t.Fatal(err)
	}
	if got := grouped.Count(allPresent).Values(); !slices.Equal(got, []int{2, 2, 1}) {
		t.Fatalf("Count(all-present nullable) = %v", got)
	}
	allNull, err := series.NewNullable(make([]int, 5), make([]bool, 5))
	if err != nil {
		t.Fatal(err)
	}
	if got := grouped.Count(allNull).Values(); !slices.Equal(got, []int{0, 0, 0}) {
		t.Fatalf("Count(all-null) = %v", got)
	}
	if got := grouped.FirstPresent(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(2), series.Some(8), series.None[int]()}) {
		t.Fatalf("FirstPresent = %#v", got)
	}
	if got := grouped.LastPresent(values).Optionals(); !reflect.DeepEqual(got, []series.Optional[int]{series.Some(4), series.Some(8), series.None[int]()}) {
		t.Fatalf("LastPresent = %#v", got)
	}
}

func TestGroupedAggregations_PreserveFloatingPointBehavior(t *testing.T) {
	frame, err := New(Column("row", []int{0, 1, 2, 3, 4, 5}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]string{"finite", "opposite", "infinite", "finite", "opposite", "infinite"}))
	values := series.New([]float64{math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.MaxFloat64, math.MaxFloat64, math.Inf(1)})
	means := grouped.Mean(values).Optionals()
	if means[0] != series.Some(math.MaxFloat64) || means[1] != series.Some(0.0) || !means[2].Valid || !math.IsInf(means[2].Value, 1) {
		t.Fatalf("Mean() = %+v, want [MaxFloat64 0 +Inf]", means)
	}

	nan := math.NaN()
	extrema := series.New([]float64{1, 2, nan, 3, 4, nan})
	minimums := grouped.Min(extrema).Optionals()
	maximums := grouped.Max(extrema).Optionals()
	if minimums[0] != series.Some(1.0) || minimums[1] != series.Some(2.0) || !minimums[2].Valid || !math.IsNaN(minimums[2].Value) {
		t.Fatalf("Min() = %+v, want [1 2 NaN]", minimums)
	}
	if maximums[0] != series.Some(3.0) || maximums[1] != series.Some(4.0) || !maximums[2].Valid || !math.IsNaN(maximums[2].Value) {
		t.Fatalf("Max() = %+v, want [3 4 NaN]", maximums)
	}
}

func TestGroupedMapTryMapAndResult_PreserveGroupOrder(t *testing.T) {
	frame, err := New(Column("value", []int{1, 2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]string{"a", "a", "b"}))
	values, _ := frame.Column[int]("value")
	mapped := grouped.Map(values, func(group series.Series[int]) (int, bool) {
		return group.Len(), true
	})
	if !slices.Equal(mapped.Values(), []int{2, 1}) || mapped.NullCount() != 0 {
		t.Fatalf("Map result = %#v", mapped.Optionals())
	}
	tryMapped, err := grouped.TryMap(values, func(group series.Series[int]) (int, bool, error) {
		value, present := group.At(0)
		return value, present, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := tryMapped.Optionals(), []series.Optional[int]{series.Some(1), series.Some(3)}; !slices.Equal(got, want) || !tryMapped.Nullable() {
		t.Fatalf("TryMap result = %#v, want %#v (nullable)", got, want)
	}

	wantErr := errors.New("stop")
	calls := 0
	errorGrouped := frame.GroupBy(series.New([]int{0, 1, 2}))
	failed, err := errorGrouped.TryMap(values, func(series.Series[int]) (int, bool, error) {
		calls++
		if calls == 2 {
			return 0, false, wantErr
		}
		return 1, true, nil
	})
	if !errors.Is(err, wantErr) || err.Error() != "group 1: stop" || calls != 2 {
		t.Fatalf("TryMap error = %v, calls = %d", err, calls)
	}
	if failed.Len() != 0 || failed.Nullable() {
		t.Fatalf("TryMap error result = {Len:%d Nullable:%t}, want zero Series", failed.Len(), failed.Nullable())
	}

	result, err := grouped.Result("group", ColumnFromSeries("total", grouped.Sum(values)))
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Names(); !slices.Equal(got, []string{"group", "total"}) {
		t.Fatalf("Result names = %v", got)
	}
	if _, resultErr := grouped.Result("group", Column("bad", []int{1})); !errors.Is(resultErr, ErrRowCount) {
		t.Fatalf("Result row error = %v", resultErr)
	}
}

func TestGroupedMapOperationsPanicOnNilFunction(t *testing.T) {
	grouped := (Frame{}).GroupBy(series.Series[int]{})
	values := series.Series[int]{}
	tests := []struct {
		name string
		call func()
	}{
		{name: "Map rejects a nil function", call: func() { grouped.Map[int, int](values, nil) }},
		{name: "TryMap rejects a nil function", call: func() { _, _ = grouped.TryMap[int, int](values, nil) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("function did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestGroupedOperationsPanicOnLengthMismatch(t *testing.T) {
	frame, err := New(Column("value", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	grouped := frame.GroupBy(series.New([]int{1, 1}))
	short := series.New([]int{1})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Sum rejects a mismatched length", call: func() { grouped.Sum(short) }},
		{name: "Mean rejects a mismatched length", call: func() { grouped.Mean(short) }},
		{name: "Min rejects a mismatched length", call: func() { grouped.Min(short) }},
		{name: "Max rejects a mismatched length", call: func() { grouped.Max(short) }},
		{name: "Count rejects a mismatched length", call: func() { grouped.Count(short) }},
		{name: "FirstPresent rejects a mismatched length", call: func() { grouped.FirstPresent(short) }},
		{name: "LastPresent rejects a mismatched length", call: func() { grouped.LastPresent(short) }},
		{name: "Map rejects a mismatched length", call: func() { grouped.Map(short, series.Sum[int]) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("function did not panic")
				}
			}()
			test.call()
		})
	}
	calls := 0
	_, err = grouped.TryMap(short, func(series.Series[int]) (int, bool, error) {
		calls++
		return 0, false, nil
	})
	if !errors.Is(err, ErrRowCount) || calls != 0 {
		t.Fatalf("TryMap() length error = %v, calls = %d; want ErrRowCount before callback", err, calls)
	}
}

func FuzzGroupingAgainstMapModel(f *testing.F) {
	f.Add([]byte{3, 0, 5, 3, 0, 7}, []byte{5, 0, 9, 17, 0, 3})
	f.Add([]byte{}, []byte{})
	f.Fuzz(func(t *testing.T, keyData, valueData []byte) {
		keyData = keyData[:min(len(keyData), 64)]
		keys := make([]series.Optional[int], len(keyData))
		values := make([]series.Optional[int], len(keyData))
		for i, value := range keyData {
			if value&1 != 0 {
				keys[i] = series.Some(int(int8(value)) / 16)
			}
			if len(valueData) > 0 {
				value = valueData[i%len(valueData)]
			} else {
				value = 0
			}
			if value&1 != 0 {
				values[i] = series.Some(int(int8(value)) / 2)
			}
		}

		rows := make([]int, len(keys))
		for i := range rows {
			rows[i] = i
		}
		frame, err := New(Column("row", rows))
		if err != nil {
			t.Fatal(err)
		}
		grouped := frame.GroupBy(series.FromOptionals(keys))

		type groupModel struct {
			key  series.Optional[int]
			rows []int
		}
		var groups []groupModel
		indexes := make(map[int]int)
		nullIndex := -1
		for row, key := range keys {
			index := nullIndex
			if key.Valid {
				var found bool
				index, found = indexes[key.Value]
				if !found {
					index = len(groups)
					indexes[key.Value] = index
					groups = append(groups, groupModel{key: key})
				}
			} else if index < 0 {
				index = len(groups)
				nullIndex = index
				groups = append(groups, groupModel{})
			}
			groups[index].rows = append(groups[index].rows, row)
		}

		wantKeys := make([]series.Optional[int], len(groups))
		wantSizes := make([]int, len(groups))
		wantSum := make([]series.Optional[int], len(groups))
		wantMean := make([]series.Optional[float64], len(groups))
		wantMin := make([]series.Optional[int], len(groups))
		wantMax := make([]series.Optional[int], len(groups))
		wantCount := make([]int, len(groups))
		wantFirst := make([]series.Optional[int], len(groups))
		wantLast := make([]series.Optional[int], len(groups))
		wantRows := make([][]int, len(groups))
		for i, group := range groups {
			wantKeys[i] = group.key
			wantSizes[i] = len(group.rows)
			wantRows[i] = group.rows
			for _, row := range group.rows {
				value := values[row]
				if !value.Valid {
					continue
				}
				if wantCount[i] == 0 {
					wantFirst[i] = value
					wantMin[i] = value
					wantMax[i] = value
				}
				wantLast[i] = value
				wantSum[i] = series.Some(wantSum[i].Value + value.Value)
				wantMin[i].Value = min(wantMin[i].Value, value.Value)
				wantMax[i].Value = max(wantMax[i].Value, value.Value)
				wantCount[i]++
			}
			if wantCount[i] > 0 {
				wantMean[i] = series.Some(float64(wantSum[i].Value) / float64(wantCount[i]))
			}
		}

		if got := grouped.Keys().Optionals(); !slices.Equal(got, wantKeys) {
			t.Fatalf("Keys() = %+v, want %+v", got, wantKeys)
		}
		if got := grouped.Sizes().Values(); !slices.Equal(got, wantSizes) {
			t.Fatalf("Sizes() = %v, want %v", got, wantSizes)
		}
		var gotRows [][]int
		for _, group := range grouped.Groups() {
			column, columnErr := group.Column[int]("row")
			if columnErr != nil {
				t.Fatal(columnErr)
			}
			gotRows = append(gotRows, column.Values())
		}
		if len(gotRows) != len(wantRows) {
			t.Fatalf("Groups() count = %d, want %d", len(gotRows), len(wantRows))
		}
		for i := range gotRows {
			if !slices.Equal(gotRows[i], wantRows[i]) {
				t.Fatalf("Groups() rows = %v, want %v", gotRows, wantRows)
			}
		}
		groupValues := series.FromOptionals(values)
		if got := grouped.Sum(groupValues).Optionals(); !slices.Equal(got, wantSum) {
			t.Fatalf("Sum() = %+v, want %+v", got, wantSum)
		}
		if got := grouped.Mean(groupValues).Optionals(); len(got) != len(wantMean) {
			t.Fatalf("Mean() length = %d, want %d", len(got), len(wantMean))
		} else {
			for i := range got {
				tolerance := 1e-12 * max(1, math.Abs(wantMean[i].Value))
				if got[i].Valid != wantMean[i].Valid || got[i].Valid && math.Abs(got[i].Value-wantMean[i].Value) > tolerance {
					t.Fatalf("Mean() = %+v, want %+v", got, wantMean)
				}
			}
		}
		if got := grouped.Min(groupValues).Optionals(); !slices.Equal(got, wantMin) {
			t.Fatalf("Min() = %+v, want %+v", got, wantMin)
		}
		if got := grouped.Max(groupValues).Optionals(); !slices.Equal(got, wantMax) {
			t.Fatalf("Max() = %+v, want %+v", got, wantMax)
		}
		if got := grouped.Count(groupValues).Values(); !slices.Equal(got, wantCount) {
			t.Fatalf("Count() = %v, want %v", got, wantCount)
		}
		if got := grouped.FirstPresent(groupValues).Optionals(); !slices.Equal(got, wantFirst) {
			t.Fatalf("FirstPresent() = %+v, want %+v", got, wantFirst)
		}
		if got := grouped.LastPresent(groupValues).Optionals(); !slices.Equal(got, wantLast) {
			t.Fatalf("LastPresent() = %+v, want %+v", got, wantLast)
		}
	})
}

type intSliceHasher struct{}

func (intSliceHasher) Hash(hash *maphash.Hash, values []int) {
	maphash.WriteComparable(hash, len(values))
	for _, value := range values {
		maphash.WriteComparable(hash, value)
	}
}

func (intSliceHasher) Equal(left, right []int) bool {
	return slices.Equal(left, right)
}

func BenchmarkGroupBy(b *testing.B) {
	const size = 10_000
	inputs := []struct {
		name string
		key  series.Series[int]
	}{
		{name: "Empty", key: series.Series[int]{}},
		{name: "OneGroup", key: series.Repeat(0, size)},
		{name: "HundredGroups", key: series.NewFunc(size, func(i int) int { return i % 100 })},
		{name: "AllUnique", key: series.NewFunc(size, func(i int) int { return i })},
		{name: "NullKeys", key: series.NewNullableFunc(size, func(i int) (int, bool) { return i % 100, i%4 != 0 })},
	}
	for _, input := range inputs {
		frame := Frame{rowCount: input.key.Len()}
		reference := func() [][]int {
			groups := make([][]int, 0)
			indexes := make(map[int]int)
			nullIndex := -1
			for row := range input.key.Len() {
				value, present := input.key.At(row)
				if !present {
					if nullIndex < 0 {
						nullIndex = len(groups)
						groups = append(groups, nil)
					}
					groups[nullIndex] = append(groups[nullIndex], row)
					continue
				}
				index, found := indexes[value]
				if !found {
					index = len(groups)
					indexes[value] = index
					groups = append(groups, nil)
				}
				groups[index] = append(groups[index], row)
			}
			return groups
		}
		want := reference()
		grouped := frame.GroupBy(input.key)
		rows, offsets := collectGroupRows(grouped.rowGroups, grouped.sizes)
		for i := range want {
			if !slices.Equal(rows[offsets[i]:offsets[i+1]], want[i]) {
				b.Fatalf("%s group %d differs from reference", input.name, i)
			}
		}
		b.Run(input.name, func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				var result Grouped[int]
				for b.Loop() {
					result = frame.GroupBy(input.key)
				}
				runtime.KeepAlive(result)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				var result [][]int
				for b.Loop() {
					result = reference()
				}
				runtime.KeepAlive(result)
			})
		})
	}
	customKeys := make([][]int, size)
	for i := range customKeys {
		customKeys[i] = []int{i % 100}
	}
	customKey := series.New(customKeys)
	customFrame := Frame{rowCount: customKey.Len()}
	customReference := func() [][]int {
		groups := make([][]int, 0)
		indexes := hashmap.New[[]int, int](intSliceHasher{}, customKey.Len())
		for row, value := range customKey.Present() {
			index, found := indexes.Get(value)
			if !found {
				index = len(groups)
				indexes.Set(value, index)
				groups = append(groups, nil)
			}
			groups[index] = append(groups[index], row)
		}
		return groups
	}
	b.Run("CustomHasher", func(b *testing.B) {
		b.Run("Optimized", func(b *testing.B) {
			b.ReportAllocs()
			var result Grouped[[]int]
			for b.Loop() {
				result = customFrame.GroupByUsing(customKey, intSliceHasher{})
			}
			runtime.KeepAlive(result)
		})
		b.Run("Reference", func(b *testing.B) {
			b.ReportAllocs()
			var result [][]int
			for b.Loop() {
				result = customReference()
			}
			runtime.KeepAlive(result)
		})
	})
}

func BenchmarkGroupedSum(b *testing.B) {
	keys := make([]int, 10_000)
	values := make([]float64, len(keys))
	for i := range keys {
		keys[i] = i % 100
		values[i] = float64(i)
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	grouped := frame.GroupBy(series.New(keys))
	value := series.New(values)
	groupRows, offsets := collectGroupRows(grouped.rowGroups, grouped.sizes)
	implementations := []struct {
		name string
		sum  func() series.Series[float64]
	}{
		{name: "Optimized", sum: func() series.Series[float64] { return grouped.Sum(value) }},
		{name: "Reference", sum: func() series.Series[float64] {
			return series.NewNullableFunc(grouped.Len(), func(group int) (float64, bool) {
				var sum float64
				present := false
				for _, row := range groupRows[offsets[group]:offsets[group+1]] {
					cell, valid := value.At(row)
					if valid {
						sum += cell
						present = true
					}
				}
				return sum, present
			})
		}},
	}
	for _, implementation := range implementations {
		b.Run(implementation.name, func(b *testing.B) {
			b.ReportAllocs()
			var result series.Series[float64]
			for b.Loop() {
				result = implementation.sum()
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkGroupedCount(b *testing.B) {
	const size = 10_000
	keys := make([]int, size)
	values := make([]int, size)
	allPresent := make([]bool, size)
	partial := make([]bool, size)
	for i := range keys {
		keys[i] = i % 100
		allPresent[i] = true
		partial[i] = i%4 != 0
	}
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	grouped := frame.GroupBy(series.New(keys))
	allPresentValues, err := series.NewNullable(values, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partialValues, err := series.NewNullable(values, partial)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := series.NewNullable(values, make([]bool, size))
	if err != nil {
		b.Fatal(err)
	}
	implementations := []struct {
		name  string
		count func(series.Series[int]) series.Series[int]
	}{
		{name: "Optimized", count: grouped.Count[int]},
		{name: "Reference", count: func(input series.Series[int]) series.Series[int] {
			counts := make([]int, grouped.Len())
			for row := range input.Present() {
				counts[grouped.rowGroups[row]]++
			}
			return series.New(counts)
		}},
	}
	for _, benchmark := range []struct {
		name   string
		values series.Series[int]
	}{
		{name: "NonNull", values: series.New(values)},
		{name: "AllPresentNullable", values: allPresentValues},
		{name: "PartialNull", values: partialValues},
		{name: "AllNull", values: allNullValues},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result series.Series[int]
					for b.Loop() {
						result = implementation.count(benchmark.values)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkGroupedTryMap(b *testing.B) {
	const size = 10_000
	values := make([]int, size)
	frame, err := New(Column("value", values))
	if err != nil {
		b.Fatal(err)
	}
	value := series.New(values)
	mapGroup := func(group series.Series[int]) (int, bool, error) {
		return group.Len(), true, nil
	}
	for _, benchmark := range []struct {
		name        string
		cardinality int
	}{
		{name: "100Groups", cardinality: 100},
		{name: "AllUnique", cardinality: size},
	} {
		keys := make([]int, size)
		for i := range keys {
			keys[i] = i % benchmark.cardinality
		}
		grouped := frame.GroupBy(series.New(keys))
		implementations := []struct {
			name   string
			mapAll func() (series.Series[int], error)
		}{
			{name: "Optimized", mapAll: func() (series.Series[int], error) {
				return grouped.TryMap(value, mapGroup)
			}},
			{name: "Reference", mapAll: func() (series.Series[int], error) {
				rows, offsets := collectGroupRows(grouped.rowGroups, grouped.sizes)
				mapped := make([]int, grouped.Len())
				present := make([]bool, grouped.Len())
				for group := range grouped.Len() {
					result, valid, mapErr := mapGroup(value.Take(rows[offsets[group]:offsets[group+1]]))
					if mapErr != nil {
						return series.Series[int]{}, mapErr
					}
					mapped[group] = result
					present[group] = valid
				}
				return series.NewNullable(mapped, present)
			}},
		}
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result series.Series[int]
					for b.Loop() {
						var mapErr error
						result, mapErr = implementation.mapAll()
						if mapErr != nil {
							b.Fatal(mapErr)
						}
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}
